"""Benchmark FA4 dropout vs PyTorch SDPA cuDNN backend.

Compares forward + backward latency / throughput across:

  * FA4 (CuTe SM100 Python kernel) without dropout
  * FA4 with dropout (``p_dropout > 0``)
  * ``torch.nn.functional.scaled_dot_product_attention`` forced onto the
    cuDNN SDPA backend, without and with dropout (which is what production
    FSDP / PyTorch training pipelines that use ``F.scaled_dot_product_attention``
    end up running on Hopper/Blackwell)

The benchmark intentionally mirrors ``benchmarks/benchmark_attn.py``: same
``do_bench`` driver, same TFLOPS / bandwidth accounting, and the same
result-table renderer.

Usage::

    python tests/cute/benchmark_dropout.py
    python tests/cute/benchmark_dropout.py --seqlen 1k,2k,4k --headdim 64,128
    python tests/cute/benchmark_dropout.py --p-dropout 0.1875 --bwd

cuDNN routes ``F.scaled_dot_product_attention`` to its fused SDPA kernel
only when ``dropout_p`` is a multiple of 1/16 (see PyTorch PR #174245);
defaults below pick rates that satisfy that constraint.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from triton.testing import do_bench

from flash_attn.cute.bench_utils import flops
from flash_attn.cute.interface import flash_attn_func as fa4_flash_attn_func


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_bwd_fn(fwd_fn, g, inputs):
    """Run fwd once and return a closure that benchmarks just the bwd pass."""
    out = fwd_fn()
    if isinstance(out, tuple):
        out = out[0]
    g_match = g[: out.shape[0]] if g.shape[0] != out.shape[0] else g

    def bwd_fn():
        for x in inputs:
            if x is not None:
                x.grad = None
        out.backward(g_match, retain_graph=True)

    return bwd_fn


# ── Backend setups ──────────────────────────────────────────────────────────


def setup_fa4(ctx, p_dropout: float):
    """FA4 forward / backward closures. ``p_dropout == 0`` exercises the
    no-dropout kernel path; ``p_dropout > 0`` exercises the dropout path with
    a stable Philox seed so cache behaviour is comparable across runs."""
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    rng_state = torch.tensor([42, 0], dtype=torch.int64) if p_dropout > 0 else None

    def fwd_fn():
        return fa4_flash_attn_func(
            q, k, v,
            causal=causal,
            p_dropout=p_dropout,
            rng_state=rng_state,
        )

    bwd_fn = None
    if ctx["has_backward"]:
        bwd_fn = _make_bwd_fn(fwd_fn, g, [q, k, v])
    return fwd_fn, bwd_fn


def setup_cudnn_sdpa(ctx, p_dropout: float):
    """PyTorch's ``F.scaled_dot_product_attention`` forced onto cuDNN.

    cuDNN expects (batch, nheads, seqlen, headdim) layout; FA's native
    layout is (batch, seqlen, nheads, headdim) so transpose before/after.
    Returns ``(None, None)`` if cuDNN cannot serve this configuration
    (e.g. older PyTorch builds without the cuDNN SDPA backend, or a
    dropout rate that the backend rejects).
    """
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    qt = q.transpose(1, 2).contiguous().detach()
    kt = k.transpose(1, 2).contiguous().detach()
    vt = v.transpose(1, 2).contiguous().detach()
    gt = g.transpose(1, 2).contiguous()
    if ctx["has_backward"]:
        qt.requires_grad_(True)
        kt.requires_grad_(True)
        vt.requires_grad_(True)

    def fwd_fn():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(
                qt, kt, vt, dropout_p=p_dropout, is_causal=causal
            )

    # Probe once to surface a missing-cuDNN-build before do_bench loops on it.
    try:
        fwd_fn()
    except Exception as exc:  # pragma: no cover - depends on PyTorch/cuDNN build
        print(f"[cuDNN SDPA] kernel not available for this config: {exc}")
        return None, None

    bwd_fn = None
    if ctx["has_backward"]:
        bwd_fn = _make_bwd_fn(fwd_fn, gt, [qt, kt, vt])
    return fwd_fn, bwd_fn


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_int_k(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("k"):
        return int(s[:-1]) * 1024
    return int(s)


def csv_ints(s: str) -> list[int]:
    return [parse_int_k(x) for x in s.split(",")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark FA4 dropout / no-dropout vs PyTorch SDPA cuDNN."
    )
    p.add_argument("--headdim", type=csv_ints, default=[128],
                   help="Head dim(s), comma-separated (default: 64,128)")
    p.add_argument("--seqlen", type=csv_ints, default=[8192, 32768],
                   help="Seq length(s), comma-separated with k suffix (default: 1k,2k,4k,8k)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size (default: total_seqlen // seqlen, min 1)")
    p.add_argument("--total-seqlen", type=parse_int_k, default="32k",
                   help="Target total tokens per benchmark for autoscaling batch (default: 16k)")
    p.add_argument("--nheads", type=int, default=None,
                   help="# Q heads (default: 16 if hdim<=128 else 8)")
    p.add_argument("--p-dropout", type=float, default=0.1,
                   help="Dropout probability (default: 0.125 = 2/16, a cuDNN-friendly rate)")
    p.add_argument("--causal", type=str.lower, choices=["true", "false", "both"], default="both",
                   help="Causal mode (default: both)")
    p.add_argument("--fwd", action="store_true",
                   help="Run forward only (default: run both fwd and bwd)")
    p.add_argument("--bwd", action="store_true",
                   help="Run backward only (default: run both fwd and bwd)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=10)
    return p.parse_args()


# ── Main loop ───────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # fwd/bwd selection:
    #   - default (neither flag): run both fwd and bwd
    #   - --fwd  only         : run forward only
    #   - --bwd  only         : run backward only
    #   - --fwd --bwd         : run both (same as default)
    if not args.fwd and not args.bwd:
        has_forward = True
        has_backward = True
    else:
        has_forward = args.fwd
        has_backward = args.bwd

    if args.causal == "true":
        causal_vals = [True]
    elif args.causal == "false":
        causal_vals = [False]
    else:
        causal_vals = [False, True]

    p_dropout = args.p_dropout
    if not (0.0 <= p_dropout < 1.0):
        raise SystemExit(f"--p-dropout must be in [0, 1), got {p_dropout}")
    # cuDNN backend silently rejects non-1/16-multiple dropout rates on some
    # PyTorch/cuDNN combos; warn the user but proceed (FA4 supports anything).
    eps = 1e-6
    if abs(p_dropout * 16 - round(p_dropout * 16)) > eps:
        print(f"[WARN] p_dropout={p_dropout} is not a multiple of 1/16; "
              f"cuDNN may fall back to flash-attention.")

    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"

    # Backend rows: (short_name, full_label, setup_fn, p_dropout_value).
    # ``short_name`` is used internally as a stable key (independent of
    # ``p_dropout``) so the post-processing table can compute speedup / overhead
    # ratios; ``full_label`` is what we print to the user.
    p_lbl = f"p={p_dropout:g}"
    backends = [
        ("fa4_off",   "FA4 (no drop)",     lambda c: setup_fa4(c, 0.0),       0.0),
        ("fa4_on",    f"FA4 ({p_lbl})",    lambda c: setup_fa4(c, p_dropout), p_dropout),
        ("cudnn_off", "cuDNN (no drop)",   lambda c: setup_cudnn_sdpa(c, 0.0), 0.0),
        ("cudnn_on",  f"cuDNN ({p_lbl})",  lambda c: setup_cudnn_sdpa(c, p_dropout), p_dropout),
    ]
    backend_short_names = [b[0] for b in backends]
    backend_labels = {b[0]: b[1] for b in backends}

    time_f: dict[tuple, float] = {}
    time_b: dict[tuple, float] = {}

    for headdim in args.headdim:
        nheads = args.nheads if args.nheads is not None else (16 if headdim <= 128 else 8)
        nheads_kv = nheads

        for seqlen in args.seqlen:
            batch_size = (
                args.batch_size
                if args.batch_size is not None
                else max(1, args.total_seqlen // seqlen)
            )

            q = torch.randn(batch_size, seqlen, nheads, headdim,
                            device=device, dtype=dtype, requires_grad=has_backward)
            k = torch.randn(batch_size, seqlen, nheads_kv, headdim,
                            device=device, dtype=dtype, requires_grad=has_backward)
            v = torch.randn(batch_size, seqlen, nheads_kv, headdim,
                            device=device, dtype=dtype, requires_grad=has_backward)
            g = torch.randn(batch_size, seqlen, nheads, headdim,
                            device=device, dtype=dtype)

            for causal in causal_vals:
                cfg = (headdim, causal, seqlen, batch_size, nheads)

                ctx = dict(
                    q=q, k=k, v=v, g=g, causal=causal,
                    headdim=headdim, has_backward=has_backward,
                )

                for short_name, full_label, setup_fn, _p in backends:
                    fwd_fn, bwd_fn = setup_fn(ctx)
                    if fwd_fn is not None and has_forward:
                        time.sleep(0.5)
                        print(f"Benchmarking {full_label} fwd, "
                              f"hdim={headdim}, seqlen={seqlen}, "
                              f"causal={causal}, batch={batch_size}, nheads={nheads}")
                        sec = do_bench(fwd_fn, warmup=args.warmup, rep=args.rep) * 1e-3
                        time_f[(cfg, short_name)] = sec
                    if bwd_fn is not None and has_backward:
                        time.sleep(0.5)
                        print(f"Benchmarking {full_label} bwd, "
                              f"hdim={headdim}, seqlen={seqlen}, "
                              f"causal={causal}, batch={batch_size}, nheads={nheads}")
                        sec = do_bench(bwd_fn, warmup=args.warmup, rep=args.rep) * 1e-3
                        time_b[(cfg, short_name)] = sec

    # ── Print results: one block per direction; two tables per block
    # (latency in microseconds + throughput in TFLOPS + derived ratios).
    _render_results(
        time_f,
        time_b,
        backend_short_names,
        backend_labels,
        p_dropout,
        has_forward,
        has_backward,
    )


# ── Result rendering ────────────────────────────────────────────────────────


def _render_results(
    time_f: dict,
    time_b: dict,
    backend_keys: list[str],
    backend_labels: dict[str, str],
    p_dropout: float,
    has_forward: bool,
    has_backward: bool,
) -> None:
    """Render benchmark results so FA4 and cuDNN sit on the same row.

    Each row corresponds to one ``(config × dropout)`` setting. The right
    half of the row is split into four backend×direction groups (each
    showing ``μs`` and ``TFLOPS``):

        FA4 fwd | FA4 bwd | cuDNN fwd | cuDNN bwd

    With this layout, side-by-side comparison between FA4 and cuDNN under
    the same workload is just reading across one line.
    """
    p_lbl = f"p={p_dropout:g}"

    # All configs that have at least one measurement (fwd or bwd).
    configs = sorted({k[0] for k in (list(time_f) + list(time_b))})
    if not configs:
        return

    # Each row is one (config × dropout); within a config we emit two rows.
    dropout_states = [
        ("off",  "off",   "fa4_off",   "cudnn_off"),
        ("on",   p_lbl,   "fa4_on",    "cudnn_on"),
    ]

    # ── Column layout ────────────────────────────────────────────────────────
    # Each metric cell shows two numbers — latency (μs) and throughput
    # (TFLOPS) — joined by " / ". Width 15 fits e.g. "1326.2 /  259.1".
    #
    # COL_SPECS: (key, top_label, bottom_label, width)
    # - top_label    : line-1 (column name or backend×direction group)
    # - bottom_label : line-2 (units; empty for config columns)
    METRIC_W = 15
    COL_SPECS = [
        ("hdim",      "hdim",      "",              4),
        ("causal",    "causal",    "",              6),
        ("batch",     "batch",     "",              5),
        ("seqlen",    "seqlen",    "",              6),
        ("nheads",    "nheads",    "",              6),
        ("dropout",   "dropout",   "",              8),
        ("fa4_fwd",   "FA4 fwd",   "    μs / TFLOPS", METRIC_W),
        ("fa4_bwd",   "FA4 bwd",   "    μs / TFLOPS", METRIC_W),
        ("cudnn_fwd", "cuDNN fwd", "    μs / TFLOPS", METRIC_W),
        ("cudnn_bwd", "cuDNN bwd", "    μs / TFLOPS", METRIC_W),
    ]

    def _row(values: dict) -> str:
        cells = [f" {str(values[k]):>{w}} " for k, _, _, w in COL_SPECS]
        return "|" + "|".join(cells) + "|"

    def _top_header_row() -> str:
        return _row({k: top for k, top, _, _ in COL_SPECS})

    def _sub_header_row() -> str:
        return _row({k: sub for k, _, sub, _ in COL_SPECS})

    def _separator(char: str = "-") -> str:
        return "+" + "+".join(char * (w + 2) for _, _, _, w in COL_SPECS) + "+"

    def _fmt_cell(t: float | None, flops_mult: float, nFLOPS: float) -> str:
        # Each cell renders as "{us:>6.1f} / {tflops:>6.1f}" → 15 chars wide.
        if t is None:
            return f"{'-':>6} / {'-':>6}"
        us = t * 1e6
        tflops = flops_mult * nFLOPS / t * 1e-12
        return f"{us:>6.1f} / {tflops:>6.1f}"

    # ── Print ────────────────────────────────────────────────────────────────
    title_bits = []
    if has_forward:
        title_bits.append("forward")
    if has_backward:
        title_bits.append("backward")
    title = (
        f" FA4 vs cuDNN dropout — {' + '.join(title_bits)} "
        f"(one row per config × dropout; each cell = latency_μs / TFLOPS)"
    )
    table_w = len(_sub_header_row())
    title_bar = "=" * max(table_w, len(title))

    print()
    print(title_bar)
    print(title)
    print(title_bar)
    print(_separator("="))
    print(_top_header_row())
    print(_sub_header_row())
    print(_separator("="))

    for i, cfg in enumerate(configs):
        headdim, causal, seqlen, batch_size, nheads = cfg
        nFLOPS = flops(
            batch_size, nheads, seqlen, seqlen, headdim, headdim, causal=causal
        )

        for _state_key, dropout_lbl, fa4_key, cudnn_key in dropout_states:
            fa4_f = time_f.get((cfg, fa4_key))   if has_forward  else None
            fa4_b = time_b.get((cfg, fa4_key))   if has_backward else None
            cdn_f = time_f.get((cfg, cudnn_key)) if has_forward  else None
            cdn_b = time_b.get((cfg, cudnn_key)) if has_backward else None

            # FA backward FLOP accounting uses the standard 2.5× factor
            # (1 fwd + dq + dk + dv ≈ 2.5 fwd-equivalent matmuls).
            print(_row({
                "hdim":      headdim,
                "causal":    str(causal),
                "batch":     batch_size,
                "seqlen":    seqlen,
                "nheads":    nheads,
                "dropout":   dropout_lbl,
                "fa4_fwd":   _fmt_cell(fa4_f, 1.0, nFLOPS),
                "fa4_bwd":   _fmt_cell(fa4_b, 2.5, nFLOPS),
                "cudnn_fwd": _fmt_cell(cdn_f, 1.0, nFLOPS),
                "cudnn_bwd": _fmt_cell(cdn_b, 2.5, nFLOPS),
            }))

        if i + 1 < len(configs):
            print(_separator("-"))
        else:
            print(_separator("="))


if __name__ == "__main__":
    main()
