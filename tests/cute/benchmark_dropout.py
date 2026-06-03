"""Benchmark FA4 dropout vs PyTorch SDPA cuDNN backend (varlen path only).

Compares forward + backward latency / throughput across:

  * FA4 (CuTe SM100 Python kernel) without dropout
  * FA4 with dropout (``p_dropout > 0``)
  * cuDNN SDPA varlen backend without and with dropout

All four backends are exercised through their **varlen** entry points. FA4
goes through ``flash_attn_varlen_func`` with ``cu_seqlens_{q,k}``. cuDNN goes
through ``torch.ops.aten._cudnn_attention_forward`` / ``..._backward``
directly — bypassing PyTorch's NestedTensor SDPA wrapper, which adds ~40 ms
of host-side overhead per call (cudnn-frontend graph build / plan select)
that masks the actual sub-millisecond cuDNN kernel time.

Because that wrapper overhead exists, we measure cuDNN's *GPU-only* time:

  * forward : capture into a CUDA graph and time the replay (host
    dispatch happens once at capture, replays are pure GPU work).
  * backward: autograd-driven backward can't be captured by CUDA graphs
    (CPU-side autograd plumbing breaks stream capture), so we instead use
    ``torch.profiler`` to extract the CUDA-total time of
    ``aten::_cudnn_attention_backward`` from a hot run.

FA4 has negligible host overhead per call, so ``do_bench`` (which measures
wall time) is already a faithful proxy for GPU time and we keep it.

Usage::

    python tests/cute/benchmark_dropout.py
    python tests/cute/benchmark_dropout.py --seqlen 1k,2k,4k --headdim 64,128
    python tests/cute/benchmark_dropout.py --p-dropout 0.1875 --bwd

cuDNN routes the fused SDPA kernel only when ``dropout_p`` is a multiple of
1/16 (see PyTorch PR #174245); defaults below pick rates that satisfy that
constraint.

Error handling: cuDNN's varlen path occasionally hits configurations where
its internal graph runs into an illegal-memory-access (rather than the
clean "graph.execute returned false" path). Once that happens the CUDA
context is unrecoverable for the remainder of the process — any further
allocation or kernel launch will fail with the same error. The benchmark
detects this case by ``torch.cuda.synchronize()``-ing after every cell and
aborts the sweep cleanly, printing partial results plus an ``[ABORT]``
line pinning down which ``(backend × config × direction)`` caused it.
Hint: rerun the failing point with ``CUDA_LAUNCH_BLOCKING=1`` if you need
to localize further (it makes the offending kernel itself raise instead
of the next synchronize).
"""

from __future__ import annotations

import argparse
import time

import torch

from triton.testing import do_bench

from flash_attn.cute.bench_utils import flops
from flash_attn.cute.interface import flash_attn_varlen_func as fa4_varlen_func


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_bwd_fn(fwd_fn, g, inputs):
    """Run fwd once and return a closure that benchmarks just the bwd pass."""
    out = fwd_fn()
    if isinstance(out, tuple):
        out = out[0]

    def bwd_fn():
        for x in inputs:
            if x is not None:
                x.grad = None
        out.backward(g, retain_graph=True)

    return bwd_fn


def _capture_cuda_graph(fn, warmup: int = 5):
    """Run ``fn`` ``warmup`` times then capture it into a CUDA graph.

    Used to time cuDNN forward — see module docstring for the motivation.
    Returns a no-arg replay closure that can be passed straight into
    ``do_bench``.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph.replay


def _is_cuda_ima(exc: BaseException) -> bool:
    """Return True if ``exc`` is a CUDA "illegal memory access" / unspecified
    launch failure / context-poisoned style error.

    After such an error the CUDA context is unusable for the rest of the
    process (no further kernels, no further sync, no further .clone()).
    The benchmark needs to abort the sweep instead of trying the next cell.
    """
    msg = str(exc).lower()
    return (
        "illegal memory access" in msg
        or "unspecified launch failure" in msg
        or ("cuda context" in msg and "destroy" in msg)
        or "device-side assert" in msg
    )


def _bench_via_profiler(fn, op_name: str, warmup: int = 5, rep: int = 20) -> float:
    """Run ``fn`` under ``torch.profiler`` and return per-call GPU time (ms).

    Filters profiler key_averages for events whose key contains ``op_name``,
    sums their ``device_time_total`` (total GPU time including children),
    then divides by the number of outer iterations. This pulls the cuDNN
    backward kernel time out from under PyTorch's host overhead.
    """
    from torch.profiler import profile, ProfilerActivity

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(rep):
            fn()
        torch.cuda.synchronize()
    total_us = 0.0
    for ev in prof.key_averages():
        if op_name in ev.key:
            # device_time_total is the SUM across all `count` occurrences.
            total_us = max(total_us, float(ev.device_time_total))
    return (total_us / rep) * 1e-3


# ── Backend setups ──────────────────────────────────────────────────────────


def setup_fa4(ctx, p_dropout: float):
    """FA4 forward / backward closures via the varlen API.

    ``q/k/v`` here are flat ``(total_q, nheads, headdim)`` tensors and
    ``cu_seqlens_*`` is precomputed in ``ctx``. With ``p_dropout==0`` the
    nodrop kernel path is exercised; with ``p_dropout>0`` the dropout
    variant is JIT-compiled. We pin a deterministic Philox seed so the
    cache key + DRAM access pattern is stable across runs.
    """
    q, k, v = ctx["q_flat"], ctx["k_flat"], ctx["v_flat"]
    g = ctx["g_flat"]
    cu_q, cu_k = ctx["cu_q"], ctx["cu_k"]
    max_q, max_k = ctx["max_q"], ctx["max_k"]
    causal = ctx["causal"]
    rng_state = torch.tensor([42, 0], dtype=torch.int64) if p_dropout > 0 else None

    def fwd_fn():
        return fa4_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=causal,
            p_dropout=p_dropout,
            rng_state=rng_state,
        )

    bwd_fn = None
    if ctx["has_backward"]:
        bwd_fn = _make_bwd_fn(fwd_fn, g, [q, k, v])
    return fwd_fn, bwd_fn


def setup_cudnn_sdpa(ctx, p_dropout: float):
    """cuDNN varlen SDPA via direct ATen calls.

    We bypass ``F.scaled_dot_product_attention(NJT)`` because PyTorch's
    NestedTensor wrapper goes through ``run_cudnn_SDP_fprop_nestedtensor`` /
    ``..._bprop_nestedtensor`` which carry ~40 ms of host-side overhead per
    call on this build (cudnn-frontend plan select + ragged-offset setup
    kernels). The actual cuDNN GPU kernel is sub-millisecond.

    Returned closures:

      * ``fwd_fn``: a CUDA-graph replay closure. Capturing strips Python /
        cudnn-frontend / caching-allocator host time; replay submits only
        the recorded GPU work.
      * ``bwd_fn``: a plain autograd-driven closure tagged with
        ``__cudnn_bench_mode__ = "profiler"``. The benchmark loop sees
        the tag and times this via ``_bench_via_profiler`` instead of
        ``do_bench`` (CUDA-graph capture of autograd backward fails with
        ``cudaErrorStreamCaptureInvalidated`` on this PyTorch).

    Returns ``(None, None)`` if cuDNN cannot serve this configuration.
    """
    causal = ctx["causal"]
    # Detach + clone so we own the autograd graph independently from FA4.
    qt = ctx["q_flat"].detach().clone()
    kt = ctx["k_flat"].detach().clone()
    vt = ctx["v_flat"].detach().clone()
    gt = ctx["g_flat"].detach().clone()
    cu = ctx["cu_q"]
    max_q = ctx["max_q"]
    max_k = ctx["max_k"]

    def fwd_raw():
        return torch.ops.aten._cudnn_attention_forward(
            qt, kt, vt,
            None,            # attn_bias (Optional in this op's schema)
            cu, cu,
            max_q, max_k,
            True,            # compute_log_sumexp
            p_dropout,
            causal,
            False,           # return_debug_mask
            scale=None,
        )

    try:
        fwd_raw()
    except Exception as exc:  # pragma: no cover - depends on PyTorch/cuDNN build
        print(f"[cuDNN SDPA] kernel not available for this config: {exc}")
        return None, None

    # ── Forward via CUDA-graph replay (GPU-only timing) ──────────────────
    fwd_fn = _capture_cuda_graph(fwd_raw, warmup=5)

    # ── Backward via autograd, timed by profiler (GPU-only) ───────────────
    bwd_fn = None
    if ctx["has_backward"]:
        qt.requires_grad_(True)
        kt.requires_grad_(True)
        vt.requires_grad_(True)

        # Build a *fresh* autograd graph once; the bwd closure will reuse it
        # so we measure only the bwd kernel time, not fwd setup time.
        out_keep, *_ = torch.ops.aten._cudnn_attention_forward(
            qt, kt, vt, None, cu, cu, max_q, max_k,
            True, p_dropout, causal, False, scale=None,
        )

        def bwd_fn():
            return torch.autograd.grad(
                outputs=[out_keep],
                inputs=[qt, kt, vt],
                grad_outputs=[gt],
                retain_graph=True,
            )

        # Probe the bwd path once so we surface "graph.execute returned false"
        # / unsupported-config / OOM failures up front instead of inside the
        # timing loop. cuDNN's varlen bwd has tighter shape constraints than
        # the fwd, so a passing fwd does not imply a passing bwd.
        try:
            bwd_fn()
        except Exception as exc:  # pragma: no cover - cuDNN/SMEM specific
            print(f"[cuDNN SDPA] backward not available for this config: {exc}")
            return fwd_fn, None

        # Tag so the benchmark loop knows to use profiler-based timing.
        bwd_fn.__cudnn_bench_mode__ = "profiler"  # type: ignore[attr-defined]
        bwd_fn.__cudnn_bench_op__ = "aten::_cudnn_attention_backward"  # type: ignore[attr-defined]

    return fwd_fn, bwd_fn


# ── Per-cell driver ─────────────────────────────────────────────────────────


def _run_backend_cell(
    *,
    short_name: str,
    full_label: str,
    setup_fn,
    ctx: dict,
    cfg: tuple,
    time_f: dict,
    time_b: dict,
    has_forward: bool,
    has_backward: bool,
    warmup: int,
    rep: int,
    headdim: int,
    seqlen: int,
    causal: bool,
    batch_size: int,
    nheads: int,
) -> bool:
    """Run one (config × backend) cell. Returns False iff the CUDA context
    was poisoned (illegal-memory-access etc.) and the whole sweep must be
    aborted; True otherwise (including normal "[skip]" outcomes where the
    backend simply doesn't support this shape).

    After each phase we explicitly ``torch.cuda.synchronize()`` so that any
    async kernel failure surfaces *here* — pinned to the offending backend
    and direction — instead of deferring to the next backend's first
    allocation, which is how the original code lost the attribution.
    """
    cfg_str = (f"hdim={headdim}, seqlen={seqlen}, causal={causal}, "
               f"batch={batch_size}, nheads={nheads}")

    # ── Setup ────────────────────────────────────────────────────────────
    try:
        fwd_fn, bwd_fn = setup_fn(ctx)
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_cuda_ima(exc):
            print(f"[ABORT] {full_label} setup hit a CUDA fatal error "
                  f"({cfg_str}): {exc}")
            return False
        # Non-fatal setup failure (e.g. cuDNN says "unsupported config").
        print(f"[skip] {full_label} setup failed: {exc}")
        return True

    # ── Forward ──────────────────────────────────────────────────────────
    if fwd_fn is not None and has_forward:
        # Pause so the GPU can clock back up before each measurement;
        # otherwise neighbouring kernels inherit the previous run's thermal
        # / DVFS state and the comparison is unfair.
        time.sleep(1.0)
        print(f"Benchmarking {full_label} fwd, {cfg_str}")
        # cuDNN fwd is a CUDA-graph replay closure (already GPU-only);
        # FA4 fwd is a direct call where wall ≈ GPU.
        try:
            sec = do_bench(fwd_fn, warmup=warmup, rep=rep) * 1e-3
            torch.cuda.synchronize()
            time_f[(cfg, short_name)] = sec
        except Exception as exc:  # pragma: no cover
            if _is_cuda_ima(exc):
                print(f"[ABORT] {full_label} fwd hit a CUDA fatal error "
                      f"({cfg_str}): {exc}")
                return False
            # e.g. cuDNN's varlen graph.execute() can fail at certain
            # (seqlen, batch, head_dim) corners with "mha_graph.execute ...
            # got false". Skip the cell rather than aborting the sweep.
            print(f"[skip] {full_label} fwd failed: {exc}")

    # ── Backward ─────────────────────────────────────────────────────────
    if bwd_fn is not None and has_backward:
        time.sleep(1.0)
        print(f"Benchmarking {full_label} bwd, {cfg_str}")
        # cuDNN bwd carries ~40ms of host-side wrapper overhead that
        # wall-clock benchmarks can't see through and that CUDA-graph
        # capture refuses (autograd plumbing breaks stream capture); fall
        # back to profiler-extracted GPU-only time for that case. FA4
        # backward is already async + fast, so do_bench is faithful.
        bench_mode = getattr(bwd_fn, "__cudnn_bench_mode__", None)
        try:
            if bench_mode == "profiler":
                op_name = bwd_fn.__cudnn_bench_op__  # type: ignore[attr-defined]
                sec = _bench_via_profiler(
                    bwd_fn, op_name, warmup=warmup, rep=rep,
                ) * 1e-3
            else:
                sec = do_bench(bwd_fn, warmup=warmup, rep=rep) * 1e-3
            torch.cuda.synchronize()
            time_b[(cfg, short_name)] = sec
        except Exception as exc:  # pragma: no cover
            if _is_cuda_ima(exc):
                print(f"[ABORT] {full_label} bwd hit a CUDA fatal error "
                      f"({cfg_str}): {exc}")
                return False
            print(f"[skip] {full_label} bwd failed: {exc}")

    return True


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
        description="Benchmark FA4 dropout / no-dropout vs PyTorch SDPA cuDNN (varlen)."
    )
    p.add_argument("--headdim", type=csv_ints, default=[64, 128],
                   help="Head dim(s), comma-separated (default: 64,128)")
    p.add_argument("--seqlen", type=csv_ints, default=[1024, 4096, 8192, 16384, 32768],
                   help="Seq length(s), comma-separated with k suffix (default: 1k,4k,8k,16k,32k)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size (default: total_seqlen // seqlen, min 1)")
    p.add_argument("--total-seqlen", type=parse_int_k, default="32k",
                   help="Target total tokens per benchmark for autoscaling batch (default: 32k)")
    p.add_argument("--nheads", type=int, default=None,
                   help="# Q heads (default: 16 if hdim<=128 else 8)")
    p.add_argument("--p-dropout", type=float, default=0.125,
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

            # All sequences share the same length; we dispatch via the varlen
            # path by giving cu_seqlens = [0, S, 2S, ...]. q/k/v are flat
            # (total, h, d) buffers consumed by both the FA4 varlen kernel
            # and PyTorch's NestedTensor wrapper.
            total = batch_size * seqlen
            cu = torch.arange(
                0, (batch_size + 1) * seqlen,
                step=seqlen, dtype=torch.int32, device=device,
            )
            q_flat = torch.randn(total, nheads, headdim,
                                 device=device, dtype=dtype, requires_grad=has_backward)
            k_flat = torch.randn(total, nheads_kv, headdim,
                                 device=device, dtype=dtype, requires_grad=has_backward)
            v_flat = torch.randn(total, nheads_kv, headdim,
                                 device=device, dtype=dtype, requires_grad=has_backward)
            g_flat = torch.randn(total, nheads, headdim,
                                 device=device, dtype=dtype)

            for causal in causal_vals:
                cfg = (headdim, causal, seqlen, batch_size, nheads)

                ctx = dict(
                    q_flat=q_flat, k_flat=k_flat, v_flat=v_flat, g_flat=g_flat,
                    cu_q=cu, cu_k=cu, max_q=seqlen, max_k=seqlen,
                    causal=causal,
                    headdim=headdim, has_backward=has_backward,
                )

                aborted = False
                for short_name, full_label, setup_fn, _p in backends:
                    # Per-backend cell. ``_run_backend_cell`` returns False
                    # iff CUDA context was poisoned (IMA) and the sweep must
                    # be aborted, otherwise True (including normal skips).
                    ok = _run_backend_cell(
                        short_name=short_name,
                        full_label=full_label,
                        setup_fn=setup_fn,
                        ctx=ctx,
                        cfg=cfg,
                        time_f=time_f,
                        time_b=time_b,
                        has_forward=has_forward,
                        has_backward=has_backward,
                        warmup=args.warmup,
                        rep=args.rep,
                        headdim=headdim,
                        seqlen=seqlen,
                        causal=causal,
                        batch_size=batch_size,
                        nheads=nheads,
                    )
                    if not ok:
                        aborted = True
                        break
                if aborted:
                    break
            if aborted:
                break
        if aborted:
            break

    if aborted:
        # CUDA context is poisoned for the rest of the process; we cannot
        # recover by clearing the cache or resetting the device. Print
        # whatever we did manage to collect so the run is not a total loss.
        print()
        print("=" * 70)
        print("Sweep aborted by a CUDA fatal error — see [ABORT] line above "
              "for the offending (backend × config). Partial results follow.")
        print("=" * 70)

    # ── Print results: one combined table with latency (ms) and throughput
    # (TFLOPS) per backend×direction cell.
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
    showing ``ms`` and ``TFLOPS``):

        FA4 fwd | cuDNN fwd | FA4 bwd | cuDNN bwd

    Forward columns are grouped together (and likewise for backward) so
    that comparing FA4 against cuDNN within a single direction is just
    glancing at two adjacent cells.
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
    # Each metric cell shows two numbers — latency (ms) and throughput
    # (TFLOPS) — joined by " / ". Width 15 fits e.g. "  1.33 /  259.1".
    #
    # COL_SPECS: (key, top_label, bottom_label, width)
    # - top_label    : line-1 (column name or backend×direction group)
    # - bottom_label : line-2 (units; empty for config columns)
    #
    # Direction-major ordering: all fwd columns first, then all bwd columns,
    # so FA4 ↔ cuDNN comparisons within a direction are visually adjacent.
    METRIC_W = 15
    COL_SPECS = [
        ("hdim",      "hdim",      "",              4),
        ("causal",    "causal",    "",              6),
        ("batch",     "batch",     "",              5),
        ("seqlen",    "seqlen",    "",              6),
        ("nheads",    "nheads",    "",              6),
        ("dropout",   "dropout",   "",              8),
        ("fa4_fwd",   "FA4 fwd",   "    ms / TFLOPS", METRIC_W),
        ("cudnn_fwd", "cuDNN fwd", "    ms / TFLOPS", METRIC_W),
        ("fa4_bwd",   "FA4 bwd",   "    ms / TFLOPS", METRIC_W),
        ("cudnn_bwd", "cuDNN bwd", "    ms / TFLOPS", METRIC_W),
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
        # Each cell renders as "{ms:>6.2f} / {tflops:>6.1f}" → 15 chars wide.
        # 2 decimal places keep sub-millisecond kernels (e.g. 0.05 ms) legible
        # while still fitting workloads up to ~999 ms in 6 chars.
        if t is None:
            return f"{'-':>6} / {'-':>6}"
        ms = t * 1e3
        tflops = flops_mult * nFLOPS / t * 1e-12
        return f"{ms:>6.2f} / {tflops:>6.1f}"

    # ── Print ────────────────────────────────────────────────────────────────
    title_bits = []
    if has_forward:
        title_bits.append("forward")
    if has_backward:
        title_bits.append("backward")
    title = (
        f" FA4 vs cuDNN dropout (varlen) — {' + '.join(title_bits)} "
        f"(one row per config × dropout; each cell = latency_ms / TFLOPS)"
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
                "cudnn_fwd": _fmt_cell(cdn_f, 1.0, nFLOPS),
                "fa4_bwd":   _fmt_cell(fa4_b, 2.5, nFLOPS),
                "cudnn_bwd": _fmt_cell(cdn_b, 2.5, nFLOPS),
            }))

        if i + 1 < len(configs):
            print(_separator("-"))
        else:
            print(_separator("="))


if __name__ == "__main__":
    main()
