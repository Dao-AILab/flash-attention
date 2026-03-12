#!/usr/bin/env python
"""Unified SM90 benchmark for forward and backward passes.

Usage:
    # Default: bench fwd+bwd for hdim 64,96,128 at seqlen 8192
    python benchmarks/bench_sm90.py

    # Forward only, specific hdims
    python benchmarks/bench_sm90.py --direction fwd --hdim 64 96

    # Backward only
    python benchmarks/bench_sm90.py --direction bwd --hdim 128

    # Custom seqlens and batch size
    python benchmarks/bench_sm90.py --seqlen 1024 2048 4096 8192 --batch 0

    # Sweep tile sizes for fwd
    python benchmarks/bench_sm90.py --sweep-tiles --hdim 96

    # Sweep tile sizes for fwd (all hdims including 192, 256)
    python benchmarks/bench_sm90.py --sweep-tiles --hdim 64 96 128 192 256

    # Sweep RS/overlap variants
    python benchmarks/bench_sm90.py --sweep-rs-overlap --hdim 64 96

    # Compare old vs new configs
    python benchmarks/bench_sm90.py --compare-configs

    # Sweep backward optimizations (V_in_regs, mma_dkv_is_rs, pipeline sharing)
    python benchmarks/bench_sm90.py --sweep-bwd-opts --hdim 64 128

    # Causal only, more reps
    python benchmarks/bench_sm90.py --causal-only --rep 50
"""
import argparse
import time

import torch
import torch.nn.functional as F
from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd


# ── Helpers ────────────────────────────────────────────────────────────────

def nheads_for_hdim(h):
    return 32 if h <= 64 else (16 if h <= 192 else 8)


def fwd_flops(batch, nheads, seqlen, hdim, hdim_v=None, causal=False):
    if hdim_v is None:
        hdim_v = hdim
    avg_seqlen = seqlen / 2 if causal else seqlen
    return batch * nheads * 2 * seqlen * avg_seqlen * (hdim + hdim_v)


def bwd_flops(batch, nheads, seqlen, hdim, causal=False):
    return 2.5 * fwd_flops(batch, nheads, seqlen, hdim, causal=causal)


def get_causals(args):
    if args.causal_only:
        return [True]
    if args.non_causal_only:
        return [False]
    return [False, True]


def auto_batch(seqlen, batch_arg, total_tokens=32768):
    return batch_arg if batch_arg > 0 else max(1, total_tokens // seqlen)


# ── Core bench functions ──────────────────────────────────────────────────

def bench_fwd(batch, seqlen, nheads, hdim, causal, tile_m=None, tile_n=None,
              mma_pv_is_rs=None, intra_wg_overlap=None, check_correctness=True,
              warmup=5, rep=30):
    """Benchmark forward pass. Returns (ms, tflops, max_diff_or_error)."""
    q = torch.randn(batch, seqlen, nheads, hdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, seqlen, nheads, hdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, seqlen, nheads, hdim, dtype=torch.bfloat16, device="cuda")
    kwargs = dict(softmax_scale=hdim ** -0.5, causal=causal)
    if tile_m is not None and tile_n is not None:
        kwargs["tile_mn"] = (tile_m, tile_n)
    if mma_pv_is_rs is not None:
        kwargs["mma_pv_is_rs"] = mma_pv_is_rs
    if intra_wg_overlap is not None:
        kwargs["intra_wg_overlap"] = intra_wg_overlap

    try:
        out, _lse = _flash_attn_fwd(q, k, v, **kwargs)
    except Exception as e:
        return None, None, str(e)[:80]

    max_diff = None
    if check_correctness:
        q_ref = q.transpose(1, 2).float()
        k_ref = k.transpose(1, 2).float()
        v_ref = v.transpose(1, 2).float()
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=causal)
        out_ref = out_ref.transpose(1, 2).to(torch.bfloat16)
        max_diff = (out.float() - out_ref.float()).abs().max().item()

    for _ in range(warmup):
        _flash_attn_fwd(q, k, v, **kwargs)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        _flash_attn_fwd(q, k, v, **kwargs)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / rep
    tflops = fwd_flops(batch, nheads, seqlen, hdim, causal=causal) / ms / 1e9
    return ms, tflops, max_diff


def bench_bwd(batch, seqlen, nheads, hdim, causal, warmup=5, rep=30, **bwd_kwargs):
    """Benchmark backward pass. Returns (ms, tflops, None_or_error)."""
    q = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=torch.bfloat16)
    softmax_scale = hdim ** -0.5
    try:
        out, lse = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal,
                                    return_lse=True)
    except Exception as e:
        return None, None, str(e)[:80]
    dout = torch.randn_like(out)

    def fn():
        _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale=softmax_scale,
                         causal=causal, **bwd_kwargs)

    try:
        fn()  # compile
    except Exception as e:
        return None, None, str(e)[:80]
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / rep
    tflops = bwd_flops(batch, nheads, seqlen, hdim, causal) / ms / 1e9
    return ms, tflops, None


# ── Preset configs ────────────────────────────────────────────────────────

# (tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap)
TILE_SWEEP_CONFIGS = {
    64: [
        (192, 192, False, True),
        (192, 192, True, True),
        (192, 128, True, True),
        (192, 128, False, True),
        (128, 128, True, True),
        (128, 192, True, True),
        (192, 96, True, True),
        (192, 96, False, True),
    ],
    96: [
        (192, 144, False, True),
        (192, 144, True, True),
        (192, 128, False, True),
        (192, 128, True, True),
        (192, 96, False, True),
        (192, 96, True, True),
        (128, 128, True, True),
        (128, 128, False, True),
    ],
    128: [
        (128, 128, True, True),
        (128, 128, False, True),
        (128, 96, True, True),
        (128, 96, False, True),
        (128, 160, True, True),
        (128, 176, True, True),
        (128, 192, True, True),
    ],
    192: [
        (128, 64, True, True),
        (128, 80, True, True),
        (128, 96, True, True),
        (128, 112, True, True),
        (128, 128, True, True),
    ],
    256: [
        (128, 48, True, True),
        (128, 64, True, True),
        (128, 80, True, True),
        (128, 96, True, True),
    ],
}

RS_OVERLAP_COMBOS = [
    (True, True, "RS+OL"),
    (True, False, "RS+noOL"),
    (False, True, "noRS+OL"),
    (False, False, "noRS+noOL"),
]

COMPARE_CONFIGS = [
    # (hdim, causal, (old_tile_m, old_tile_n, old_rs, old_ol), (new...))
    (64, False, (192, 128, True, True), (192, 128, True, True)),
    (64, True, (192, 128, True, True), (192, 128, True, True)),
    (96, False, (192, 96, True, True), (192, 144, False, True)),
    (96, True, (192, 96, True, True), (192, 128, False, True)),
]


def _get_default_bwd_config(headdim, causal=False):
    """Default SM90 backward config for a given headdim."""
    if headdim <= 128:
        return dict(
            m_block_size=64 if causal else 80,
            n_block_size=128,
            num_stages_Q=2,
            num_stages_dO=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=not causal,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=1,
            num_threads=384,
        )
    elif headdim <= 192:
        return dict(
            m_block_size=64,
            n_block_size=96,
            num_stages_Q=1,
            num_stages_dO=1,
            SdP_swapAB=False,
            dKV_swapAB=True,
            dQ_swapAB=True,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=1,
            AtomLayoutMdQ=1,
            num_threads=512,
        )
    else:
        return dict(
            m_block_size=64,
            n_block_size=64,
            num_stages_Q=1,
            num_stages_dO=1,
            SdP_swapAB=False,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=1,
            AtomLayoutMdQ=1,
            num_threads=384,
        )


# Maps optimization name -> function(headdim, causal) -> dict[label, kwargs] or None
BWD_OPT_CONFIGS = {
    "V_in_regs": lambda hdim, causal: (
        None if hdim > 128 else {
            "baseline (V_in_regs=False)": {**_get_default_bwd_config(hdim, causal), "V_in_regs": False},
            "optimized (V_in_regs=True)": {**_get_default_bwd_config(hdim, causal), "V_in_regs": True},
        }
    ),
    "mma_dkv_is_rs": lambda hdim, causal: (
        None if hdim > 128 else {
            "baseline (AtomLayoutNdKV=1)": {**_get_default_bwd_config(hdim, causal), "AtomLayoutNdKV": 1},
            "optimized (AtomLayoutNdKV=2)": {**_get_default_bwd_config(hdim, causal), "AtomLayoutNdKV": 2},
        }
    ),
    "Q_dO_pipeline_sharing": lambda hdim, causal: (
        None if hdim > 128 else {
            "baseline (dO_stage=1, separate)": {**_get_default_bwd_config(hdim, causal), "num_stages_dO": 1},
            "optimized (dO_stage=2, shared)": {**_get_default_bwd_config(hdim, causal), "num_stages_dO": 2},
        }
    ),
    "tile_m": lambda hdim, causal: (
        None if hdim > 128 or causal else {
            "tile_m=64": {**_get_default_bwd_config(hdim, causal), "m_block_size": 64},
            "tile_m=80": {**_get_default_bwd_config(hdim, causal), "m_block_size": 80},
        }
    ),
}


# ── Run modes ─────────────────────────────────────────────────────────────

def run_default(args):
    """Standard fwd/bwd benchmark across hdims."""
    directions = [args.direction] if args.direction != "both" else ["fwd", "bwd"]

    for direction in directions:
        print(f"\n{'=' * 80}")
        print(f"  SM90 {direction.upper()}  (rep={args.rep})")
        print(f"{'=' * 80}")
        cols = f"{'hdim':>5} {'causal':>6} {'batch':>5} {'seqlen':>6} {'ms':>8} {'TFLOPS':>8}"
        if direction == "fwd":
            cols += f" {'max_diff':>10}"
        print(cols)
        print("-" * 80)

        for hdim in args.hdim:
            nheads = nheads_for_hdim(hdim)
            for seqlen in args.seqlen:
                batch = auto_batch(seqlen, args.batch)
                for causal in get_causals(args):
                    if direction == "fwd":
                        ms, tflops, diff = bench_fwd(batch, seqlen, nheads, hdim, causal, rep=args.rep)
                    else:
                        ms, tflops, diff = bench_bwd(batch, seqlen, nheads, hdim, causal, rep=args.rep)

                    if ms is not None:
                        line = f"{hdim:>5} {str(causal):>6} {batch:>5} {seqlen:>6} {ms:>8.3f} {tflops:>8.1f}"
                        if diff is not None:
                            line += f" {diff:>10.6f}"
                        print(line)
                    else:
                        print(f"{hdim:>5} {str(causal):>6} {batch:>5} {seqlen:>6} {'FAIL':>8} {'':>8} {diff}")


def run_sweep_tiles(args):
    """Sweep tile sizes for fwd across seqlens."""
    seqlens = args.seqlen

    for hdim in args.hdim:
        nheads = nheads_for_hdim(hdim)
        configs = TILE_SWEEP_CONFIGS.get(hdim, [])
        if not configs:
            print(f"No tile sweep configs for hdim={hdim}, skipping")
            continue

        for causal in get_causals(args):
            header = f"{'hdim':>5} {'causal':>6} {'tile_m':>6} {'tile_n':>6} {'pv_rs':>5} {'ol':>5}"
            for sl in seqlens:
                header += f" {'s=' + str(sl):>8}"
            print(header)
            print("=" * len(header))

            for tile_m, tile_n, rs, ol in configs:
                row = f"{hdim:>5} {str(causal):>6} {tile_m:>6} {tile_n:>6} {str(rs):>5} {str(ol):>5}"
                for sl in seqlens:
                    batch = auto_batch(sl, args.batch)
                    ms, tflops, diff = bench_fwd(batch, sl, nheads, hdim, causal,
                                                 tile_m, tile_n, rs, ol,
                                                 check_correctness=False, rep=args.rep)
                    row += f" {tflops:>8.1f}" if tflops else f" {'FAIL':>8}"
                print(row)
            print()


def run_sweep_rs_overlap(args):
    """Sweep RS and intra-WG-overlap combinations for fwd."""
    seqlens = args.seqlen
    tile_for_hdim = {64: (192, 128), 96: (192, 128), 128: (128, 128)}

    for hdim in args.hdim:
        nheads = nheads_for_hdim(hdim)
        tile_m, tile_n = tile_for_hdim.get(hdim, (128, 128))

        for causal in get_causals(args):
            c_str = "causal" if causal else "non-causal"
            header = f"{'Config':<30} {'RS/OL':<12}"
            for sl in seqlens:
                header += f" {'s=' + str(sl):>8}"
            print(header)
            print("=" * len(header))

            for rs, ol, rs_label in RS_OVERLAP_COMBOS:
                label = f"hdim{hdim} {c_str} {tile_m}x{tile_n}"
                row = f"{label:<30} {rs_label:<12}"
                for sl in seqlens:
                    batch = auto_batch(sl, args.batch)
                    ms, tflops, diff = bench_fwd(batch, sl, nheads, hdim, causal,
                                                 tile_m, tile_n, rs, ol,
                                                 check_correctness=False, rep=args.rep)
                    row += f" {tflops:>8.1f}" if tflops else f" {'FAIL':>8}"
                print(row)
            print()


def run_compare_configs(args):
    """Compare old vs new tile configs for fwd."""
    seqlens = args.seqlen

    header = f"{'Config':<50}"
    for sl in seqlens:
        header += f" {'s=' + str(sl):>8}"
    print(header)
    print("=" * len(header))

    for hdim, causal, old, new in COMPARE_CONFIGS:
        nheads = nheads_for_hdim(hdim)
        c_str = "causal" if causal else "non-causal"
        for label_prefix, cfg in [("OLD", old), ("NEW", new)]:
            label = f"hdim{hdim} {c_str:<11} {label_prefix}  {cfg[0]}x{cfg[1]} RS={cfg[2]} OL={cfg[3]}"
            row = f"{label:<50}"
            for sl in seqlens:
                batch = auto_batch(sl, args.batch)
                ms, tflops, diff = bench_fwd(batch, sl, nheads, hdim, causal, *cfg,
                                             check_correctness=False, rep=args.rep)
                row += f" {tflops:>8.1f}" if tflops else f" {'FAIL':>8}"
            print(row)
        print("-" * len(header))


def run_sweep_bwd_opts(args):
    """Sweep backward kernel optimizations (V_in_regs, mma_dkv_is_rs, etc.)."""
    seqlens = args.seqlen

    for opt_name, get_configs_fn in BWD_OPT_CONFIGS.items():
        for causal in get_causals(args):
            c_str = "causal" if causal else "non-causal"
            has_any = False

            for hdim in args.hdim:
                configs = get_configs_fn(hdim, causal)
                if configs is None:
                    continue
                if not has_any:
                    print(f"\n{'=' * 70}")
                    print(f"BWD Optimization: {opt_name} ({c_str})")
                    print(f"{'=' * 70}")
                    has_any = True

                nheads = nheads_for_hdim(hdim)
                print(f"\n  hdim={hdim}:")
                for sl in seqlens:
                    batch = auto_batch(sl, args.batch)
                    f = bwd_flops(batch, nheads, sl, hdim, causal)
                    if len(seqlens) > 1:
                        print(f"    seqlen={sl}, batch={batch}:")
                    for label, kwargs in configs.items():
                        ms, tflops, err = bench_bwd(batch, sl, nheads, hdim, causal,
                                                     rep=args.rep, **kwargs)
                        if ms is not None:
                            print(f"    {label:40s}: {ms:6.2f} ms  ({tflops:6.1f} TFLOPS)")
                        else:
                            print(f"    {label:40s}: FAIL  {err}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified SM90 attention benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--direction", choices=["fwd", "bwd", "both"], default="both",
                        help="Benchmark direction (default: both)")
    parser.add_argument("--hdim", type=int, nargs="+", default=[64, 96, 128],
                        help="Head dimensions (default: 64 96 128)")
    parser.add_argument("--seqlen", type=int, nargs="+", default=[8192],
                        help="Sequence lengths (default: 8192)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch size (0 = auto ~32k tokens)")
    parser.add_argument("--rep", type=int, default=30,
                        help="Repetitions per benchmark (default: 30)")
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--non-causal-only", action="store_true")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sweep-tiles", action="store_true",
                      help="Sweep fwd tile sizes")
    mode.add_argument("--sweep-rs-overlap", action="store_true",
                      help="Sweep fwd RS/overlap combos")
    mode.add_argument("--compare-configs", action="store_true",
                      help="Compare old vs new fwd tile configs")
    mode.add_argument("--sweep-bwd-opts", action="store_true",
                      help="Sweep bwd optimizations (V_in_regs, mma_dkv_is_rs, etc.)")

    args = parser.parse_args()
    torch.manual_seed(0)

    if args.sweep_tiles:
        run_sweep_tiles(args)
    elif args.sweep_rs_overlap:
        run_sweep_rs_overlap(args)
    elif args.compare_configs:
        run_compare_configs(args)
    elif args.sweep_bwd_opts:
        run_sweep_bwd_opts(args)
    else:
        run_default(args)


if __name__ == "__main__":
    main()
