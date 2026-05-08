#!/usr/bin/env python
"""SM100 Blackwell head_dim=256 benchmark (forward + backward).

Benchmarks the new 2CTA kernels added in PR #2412.  Uses the unified
`_flash_attn_fwd` / `_flash_attn_bwd` API which auto-routes to the
hd256 2CTA path on SM100/SM110 when head_dim=256.

Usage:
    # Default: fwd + bwd, seqlens 1k–16k, causal + non-causal
    python benchmarks/bench_sm100_hd256.py

    # Forward only
    python benchmarks/bench_sm100_hd256.py --direction fwd

    # Backward only
    python benchmarks/bench_sm100_hd256.py --direction bwd

    # Custom seqlens / batch
    python benchmarks/bench_sm100_hd256.py --seqlen 2048,4096,8192 --batch 2

    # Causal only
    python benchmarks/bench_sm100_hd256.py --causal-only

    # Compare FA hd256 vs PyTorch SDPA baseline
    python benchmarks/bench_sm100_hd256.py --compare-sdpa

    # Compile kernels without running (for two-pass workflow)
    python benchmarks/bench_sm100_hd256.py --compile-only

Two-pass workflow (compile in parallel, then run):
    FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \\
        python benchmarks/bench_sm100_hd256.py --compile-only

    FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \\
        python benchmarks/bench_sm100_hd256.py
"""
import argparse
import contextlib
import io
import math
import os
import sys

import torch
import torch.nn.functional as F

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd

try:
    # Preferred location in this repo.
    from benchmarks.benchmark_attn import get_peak_flops
except ModuleNotFoundError:
    # Fallback for older layouts.
    from hopper.benchmark_attn import get_peak_flops


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Suppress both Python-level (sys.stdout/err) and C-level (fd 1/2) output.

    Kernel debug prints (e.g. 'H>> shared_storage.size_in_bytes()') are emitted
    via Python print() during JIT compilation.  Redirecting only the fd is not
    enough because Python buffers writes through sys.stdout; we must also swap
    the Python stream objects.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    old_py_stdout = sys.stdout
    old_py_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        # Restore Python streams first so subsequent prints go to real stdout
        sys.stdout = old_py_stdout
        sys.stderr = old_py_stderr
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


# ── Helpers ────────────────────────────────────────────────────────────────

HEAD_DIM = 256
# Typical number of heads for head_dim=256 (matches newer model variants)
NHEADS = 8
NHEADS_KV = 8  # MHA; adjust to test GQA


def csv_ints(s):
    return [int(x.strip()) for x in s.split(",")]


def auto_batch(seqlen, batch_arg, total_tokens=32768):
    return batch_arg if batch_arg > 0 else max(1, total_tokens // seqlen)


def fwd_flops(batch, nheads, seqlen, hdim, causal=False):
    avg_seqlen = seqlen / 2 if causal else seqlen
    return batch * nheads * 2 * seqlen * avg_seqlen * (hdim + hdim)


def bwd_flops(batch, nheads, seqlen, hdim, causal=False):
    return 2.5 * fwd_flops(batch, nheads, seqlen, hdim, causal=causal)


def check_sm100():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found.", file=sys.stderr)
        sys.exit(1)
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    peak = get_peak_flops(0, dtype=torch.bfloat16)
    peak_str = f"  peak_bf16={peak/1e12:.0f} TFLOPS" if peak else ""
    if cap[0] not in (10, 11):
        print(
            f"WARNING: This benchmark targets SM100/SM110 (Blackwell). "
            f"Current GPU: {name} (SM{cap[0]}{cap[1]}). "
            f"The hd256 2CTA kernel may not be selected.",
            file=sys.stderr,
        )
    else:
        print(f"GPU: {name}  (SM{cap[0]}{cap[1]}){peak_str}")
    return peak


# ── Core bench functions ────────────────────────────────────────────────────

def bench_fwd(batch, seqlen, nheads, nheads_kv, causal,
              check_correctness=True, warmup=5, rep=30):
    """Benchmark hd256 forward pass. Returns (ms, tflops, max_diff_or_error)."""
    q = torch.randn(batch, seqlen, nheads,    HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, seqlen, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, seqlen, nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    scale = HEAD_DIM ** -0.5

    try:
        out, _lse = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal)
    except Exception as e:
        return None, None, str(e)[:120]

    max_diff = None
    if check_correctness:
        # Expand KV heads if GQA
        gqa = nheads // nheads_kv
        q_ref = q.transpose(1, 2).float()
        k_ref = k.transpose(1, 2).float().repeat_interleave(gqa, dim=1)
        v_ref = v.transpose(1, 2).float().repeat_interleave(gqa, dim=1)
        out_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=causal, scale=scale
        )
        out_ref = out_ref.transpose(1, 2).to(torch.bfloat16)
        max_diff = (out.float() - out_ref.float()).abs().max().item()

    for _ in range(warmup):
        _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    tflops = fwd_flops(batch, nheads, seqlen, HEAD_DIM, causal=causal) / ms / 1e9
    return ms, tflops, max_diff


def bench_bwd(batch, seqlen, nheads, nheads_kv, causal,
              check_correctness=True, warmup=5, rep=30):
    """Benchmark hd256 backward pass. Returns (ms, tflops, (dq_err, dk_err, dv_err) or error str)."""
    q = torch.randn(batch, seqlen, nheads,    HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    scale = HEAD_DIM ** -0.5

    try:
        out, lse = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)
    except Exception as e:
        return None, None, str(e)[:120]

    dout = torch.randn_like(out)

    def fn():
        return _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale=scale, causal=causal)

    try:
        with _suppress_stdout_stderr():
            dq, dk, dv = fn()  # compile / warm JIT — suppresses kernel debug prints
    except Exception as e:
        return None, None, str(e)[:120]

    # Gradient correctness vs PyTorch reference
    grad_errs = None
    if check_correctness:
        gqa = nheads // nheads_kv
        q_ref = q.float().detach().requires_grad_(True)
        k_ref = k.float().detach().requires_grad_(True)
        v_ref = v.float().detach().requires_grad_(True)
        k_exp = k_ref.transpose(1, 2).repeat_interleave(gqa, dim=1)
        v_exp = v_ref.transpose(1, 2).repeat_interleave(gqa, dim=1)
        out_ref = F.scaled_dot_product_attention(
            q_ref.transpose(1, 2), k_exp, v_exp, is_causal=causal, scale=scale
        ).transpose(1, 2)
        out_ref.backward(dout.float())
        dq_err = (dq.float() - q_ref.grad).abs().max().item()
        # dK/dV reference grads are summed over GQA groups; take mean per KV head
        dk_ref = k_ref.grad
        dv_ref = v_ref.grad
        dk_err = (dk.float() - dk_ref).abs().max().item()
        dv_err = (dv.float() - dv_ref).abs().max().item()
        grad_errs = (dq_err, dk_err, dv_err)

    with _suppress_stdout_stderr():
        for _ in range(warmup):
            fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    with _suppress_stdout_stderr():
        for _ in range(rep):
            fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    tflops = bwd_flops(batch, nheads, seqlen, HEAD_DIM, causal=causal) / ms / 1e9
    return ms, tflops, grad_errs


def bench_sdpa_fwd(batch, seqlen, nheads, nheads_kv, causal, warmup=5, rep=30):
    """PyTorch SDPA baseline for forward."""
    scale = HEAD_DIM ** -0.5
    gqa = nheads // nheads_kv
    q = torch.randn(batch, nheads,    seqlen, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, nheads_kv, seqlen, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, nheads_kv, seqlen, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = k.repeat_interleave(gqa, dim=1)
    v = v.repeat_interleave(gqa, dim=1)

    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    tflops = fwd_flops(batch, nheads, seqlen, HEAD_DIM, causal=causal) / ms / 1e9
    return ms, tflops


def bench_sdpa_bwd(batch, seqlen, nheads, nheads_kv, causal, warmup=5, rep=30):
    """PyTorch SDPA baseline for backward (fwd+bwd via autograd)."""
    scale = HEAD_DIM ** -0.5
    gqa = nheads // nheads_kv

    def make_inputs():
        q = torch.randn(batch, nheads,    seqlen, HEAD_DIM, dtype=torch.bfloat16,
                        device="cuda", requires_grad=True)
        k = torch.randn(batch, nheads_kv, seqlen, HEAD_DIM, dtype=torch.bfloat16,
                        device="cuda").repeat_interleave(gqa, dim=1).requires_grad_(True)
        v = torch.randn(batch, nheads_kv, seqlen, HEAD_DIM, dtype=torch.bfloat16,
                        device="cuda").repeat_interleave(gqa, dim=1).requires_grad_(True)
        return q, k, v

    q, k, v = make_inputs()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    dout = torch.randn_like(out)

    def fn():
        q_, k_, v_ = make_inputs()
        o = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal, scale=scale)
        o.backward(dout)

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    tflops = bwd_flops(batch, nheads, seqlen, HEAD_DIM, causal=causal) / ms / 1e9
    return ms, tflops


# ── Formatting helpers ─────────────────────────────────────────────────────

def fmt_tflops_mfu(tflops, peak_flops, width=18):
    """Format TFLOPS with optional MFU% as a single string, e.g. '1373.3(61.0%)'."""
    if peak_flops is not None:
        mfu = tflops * 1e12 / peak_flops * 100
        cell = f"{tflops:.1f}({mfu:.1f}%)"
    else:
        cell = f"{tflops:.1f}"
    return f"{cell:>{width}}"


# ── Run modes ──────────────────────────────────────────────────────────────

def run_default(args, peak_flops=None):
    directions = ["fwd", "bwd"] if args.direction == "both" else [args.direction]
    causals = [True] if args.causal_only else ([False] if args.non_causal_only else [False, True])
    has_mfu = peak_flops is not None

    for direction in directions:
        dir_label = "Forward" if direction == "fwd" else "Backward"

        tflops_col = "FA4 TFLOPS(MFU%)" if has_mfu else "Throughput (TFLOPS)"
        tflops_w = max(len(tflops_col), 18)

        if direction == "fwd":
            hdr = (f"{'Config (attn-mask / seqlen)':<30} {'Batch':>6} "
                   f"{'Latency (ms)':>14} {tflops_col:>{tflops_w}} {'Max Abs Err (bf16)':>19}")
        else:
            hdr = (f"{'Config (attn-mask / seqlen)':<30} {'Batch':>6} "
                   f"{'Latency (ms)':>14} {tflops_col:>{tflops_w}} "
                   f"{'dQ Err':>10} {'dK Err':>10} {'dV Err':>10}")

        width = len(hdr)
        print(f"\n{'=' * width}")
        print(f"  SM100 Blackwell  head_dim=256  2CTA  {dir_label}  "
              f"nheads={args.nheads}  nheads_kv={args.nheads_kv}  (rep={args.rep})")
        print(f"{'=' * width}")
        print(hdr)
        print("-" * width)

        for seqlen in args.seqlen:
            batch = auto_batch(seqlen, args.batch)
            for causal in causals:
                mask_label = "causal" if causal else "non-causal"
                row_name = f"{mask_label} / seqlen={seqlen}"

                if direction == "fwd":
                    ms, tflops, diff = bench_fwd(
                        batch, seqlen, args.nheads, args.nheads_kv, causal,
                        warmup=args.warmup, rep=args.rep,
                    )
                    if ms is not None:
                        line = (f"{row_name:<30} {batch:>6} {ms:>14.3f} "
                                f"{fmt_tflops_mfu(tflops, peak_flops, tflops_w)}")
                        if diff is not None:
                            line += f" {diff:>19.6f}"
                        print(line)
                    else:
                        print(f"{row_name:<30} {batch:>6} {'FAIL':>14}  {diff}")
                else:
                    ms, tflops, grad_errs = bench_bwd(
                        batch, seqlen, args.nheads, args.nheads_kv, causal,
                        warmup=args.warmup, rep=args.rep,
                    )
                    if ms is not None:
                        line = (f"{row_name:<30} {batch:>6} {ms:>14.3f} "
                                f"{fmt_tflops_mfu(tflops, peak_flops, tflops_w)}")
                        if grad_errs is not None:
                            dq_e, dk_e, dv_e = grad_errs
                            line += f" {dq_e:>10.6f} {dk_e:>10.6f} {dv_e:>10.6f}"
                        print(line)
                    else:
                        print(f"{row_name:<30} {batch:>6} {'FAIL':>14}  {grad_errs}")


def run_compare_sdpa(args, peak_flops=None):
    """Compare FA hd256 forward vs PyTorch SDPA."""
    causals = [True] if args.causal_only else ([False] if args.non_causal_only else [False, True])
    has_mfu = peak_flops is not None

    tflops_col = "FA4 TFLOPS(MFU%)" if has_mfu else "FA TFLOPS"
    tflops_w = max(len(tflops_col), 18)

    hdr = (f"{'Config (attn-mask / seqlen)':<30} {'Batch':>6} "
           f"{'FA Latency (ms)':>16} {tflops_col:>{tflops_w}} "
           f"{'SDPA Latency (ms)':>18} {'SDPA TFLOPS':>12} {'Speedup':>8}")
    width = len(hdr)

    print(f"\n{'=' * width}")
    print(f"  SM100 Blackwell  head_dim=256  Forward:  FA 2CTA  vs  PyTorch SDPA  "
          f"nheads={args.nheads}  nheads_kv={args.nheads_kv}  (rep={args.rep})")
    print(f"{'=' * width}")
    print(hdr)
    print("-" * width)

    for seqlen in args.seqlen:
        batch = auto_batch(seqlen, args.batch)
        for causal in causals:
            mask_label = "causal" if causal else "non-causal"
            row_name = f"{mask_label} / seqlen={seqlen}"

            fa_ms, fa_tflops, diff = bench_fwd(
                batch, seqlen, args.nheads, args.nheads_kv, causal,
                check_correctness=False, warmup=args.warmup, rep=args.rep,
            )
            sdpa_ms, sdpa_tflops = bench_sdpa_fwd(
                batch, seqlen, args.nheads, args.nheads_kv, causal,
                warmup=args.warmup, rep=args.rep,
            )
            if fa_ms is not None:
                speedup = sdpa_ms / fa_ms
                line = (f"{row_name:<30} {batch:>6} {fa_ms:>16.3f} "
                        f"{fmt_tflops_mfu(fa_tflops, peak_flops, tflops_w)} "
                        f"{sdpa_ms:>18.3f} {sdpa_tflops:>12.1f} {speedup:>7.2f}x")
                print(line)
            else:
                print(f"{row_name:<30} {batch:>6} {'FAIL':>16}  {fa_tflops}")


def run_compare_baseline(args, peak_flops=None):
    """Full fwd+bwd comparison: FA hd256 2CTA vs PyTorch SDPA (the only viable baseline,
    since FA4 main does not support head_dim=256 on SM100).
    """
    causals = [True] if args.causal_only else ([False] if args.non_causal_only else [False, True])
    has_mfu = peak_flops is not None

    for direction, flops_fn, fa_bench_fn, sdpa_bench_fn in [
        ("Forward",  fwd_flops, bench_fwd,
         lambda b, s, nh, nhkv, c, **kw: bench_sdpa_fwd(b, s, nh, nhkv, c, **kw)),
        ("Backward", bwd_flops, bench_bwd,
         lambda b, s, nh, nhkv, c, **kw: bench_sdpa_bwd(b, s, nh, nhkv, c, **kw)),
    ]:
        tflops_col = "FA4 TFLOPS(MFU%)" if has_mfu else "FA TFLOPS"
        tflops_w = max(len(tflops_col), 18)

        hdr = (f"{'Config (attn-mask / seqlen)':<30} {'Batch':>6} "
               f"{'FA ms':>8} {tflops_col:>{tflops_w}} "
               f"{'SDPA ms':>9} {'SDPA TFLOPS':>12} {'Speedup':>8}")
        width = len(hdr)

        print(f"\n{'=' * width}")
        print(f"  FA hd256 2CTA  vs  PyTorch SDPA  [{direction}]  "
              f"nheads={args.nheads}  nheads_kv={args.nheads_kv}  (rep={args.rep})")
        print(f"  NOTE: FA4 main does not support head_dim=256; SDPA is the only baseline.")
        print(f"{'=' * width}")
        print(hdr)
        print("-" * width)

        for seqlen in args.seqlen:
            batch = auto_batch(seqlen, args.batch)
            for causal in causals:
                mask_label = "causal" if causal else "non-causal"
                row_name = f"{mask_label} / seqlen={seqlen}"

                fa_ms, fa_tflops, _ = fa_bench_fn(
                    batch, seqlen, args.nheads, args.nheads_kv, causal,
                    check_correctness=False, warmup=args.warmup, rep=args.rep,
                )
                sdpa_ms, sdpa_tflops = sdpa_bench_fn(
                    batch, seqlen, args.nheads, args.nheads_kv, causal,
                    warmup=args.warmup, rep=args.rep,
                )

                if fa_ms is not None:
                    speedup = sdpa_ms / fa_ms
                    line = (f"{row_name:<30} {batch:>6} {fa_ms:>8.3f} "
                            f"{fmt_tflops_mfu(fa_tflops, peak_flops, tflops_w)} "
                            f"{sdpa_ms:>9.3f} {sdpa_tflops:>12.1f} {speedup:>7.2f}x")
                    print(line)
                else:
                    print(f"{row_name:<30} {batch:>6} {'FAIL':>8}  {fa_tflops}")


def run_compile_only(args):
    """Trigger JIT compilation without timing — useful for two-pass workflow."""
    causals = [False, True]
    print("Compiling hd256 2CTA kernels (fwd + bwd) ...")
    for seqlen in args.seqlen:
        batch = auto_batch(seqlen, args.batch)
        for causal in causals:
            # fwd
            q = torch.randn(batch, seqlen, args.nheads,    HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(batch, seqlen, args.nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(batch, seqlen, args.nheads_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            scale = HEAD_DIM ** -0.5
            try:
                out, lse = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)
                dout = torch.randn_like(out)
                _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale=scale, causal=causal)
                print(f"  compiled  causal={causal}  seqlen={seqlen}  batch={batch}")
            except Exception as e:
                print(f"  FAILED    causal={causal}  seqlen={seqlen}  batch={batch}  {e}")
    print("Done.")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SM100 head_dim=256 2CTA attention benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--direction", choices=["fwd", "bwd", "both"], default="both")
    parser.add_argument(
        "--seqlen", type=csv_ints, default=[1024, 2048, 4096, 8192, 16384, 32768],
        help="Comma-separated sequence lengths (default: 1024,2048,4096,8192,16384,32768)",
    )
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch size (0 = auto ~64k tokens)")
    parser.add_argument("--nheads",    type=int, default=NHEADS)
    parser.add_argument("--nheads-kv", type=int, default=NHEADS_KV, dest="nheads_kv")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep",    type=int, default=30)
    parser.add_argument("--causal-only",     action="store_true")
    parser.add_argument("--non-causal-only", action="store_true")
    parser.add_argument("--compare-sdpa",     action="store_true",
                        help="Compare FA hd256 fwd vs PyTorch SDPA")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Full fwd+bwd comparison vs PyTorch SDPA (the only viable "
                             "baseline since FA4 main does not support head_dim=256)")
    parser.add_argument("--compile-only",     action="store_true",
                        help="Compile kernels without benchmarking (two-pass step 1)")

    args = parser.parse_args()
    torch.manual_seed(0)
    peak_flops = check_sm100()

    if args.compile_only:
        run_compile_only(args)
    elif args.compare_baseline:
        run_compare_baseline(args, peak_flops=peak_flops)
    elif args.compare_sdpa:
        run_compare_sdpa(args, peak_flops=peak_flops)
    else:
        run_default(args, peak_flops=peak_flops)


if __name__ == "__main__":
    main()

