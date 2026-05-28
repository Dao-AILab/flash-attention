# Copyright (c) 2025, Tri Dao.

import argparse
import json
from pathlib import Path

import torch
from triton.testing import do_bench

from flash_attn.cute.interface import flash_attn_func


DECODE_CASES = (
    (4096, 32),
    (8192, 16),
    (16384, 8),
    (32768, 4),
)
HDIM_PAIRS = (
    (64, 64),
    (96, 96),
    (128, 128),
)


def _make_inputs(batch, seqlen_k, headdim, headdim_v, dtype, device):
    nheads = 64
    nheads_kv = 8
    q = torch.randn(batch, 1, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype)
    return q, k, v


def _benchmark_case(
    *,
    seqlen_k,
    batch,
    headdim,
    headdim_v,
    causal,
    dtype,
    warmup,
    rep,
    enable_sm100_decode_q1_opt,
    device,
):
    q, k, v = _make_inputs(batch, seqlen_k, headdim, headdim_v, dtype, device)

    def fn():
        return flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            num_splits=1,
            pack_gqa=True,
            enable_sm100_decode_q1_opt=enable_sm100_decode_q1_opt,
        )[0]

    # Compile before timing so the result reflects kernel runtime, not JIT latency.
    fn()
    torch.cuda.synchronize()
    ms = do_bench(fn, warmup=warmup, rep=rep)
    return ms


def _load_baseline(path):
    if path is None:
        return {}
    with Path(path).open() as f:
        rows = json.load(f)
    return {
        (
            int(row["headdim"]),
            int(row["headdim_v"]),
            bool(row["causal"]),
            int(row["seqlen_k"]),
            int(row["batch"]),
            bool(row["enable_sm100_decode_q1_opt"]),
        ): float(row["ms"])
        for row in rows
    }


def _format_delta(ms, baseline_ms):
    if baseline_ms is None:
        return ""
    pct = (ms / baseline_ms - 1.0) * 100.0
    return f"{pct:+.1f}%"


def main():
    parser = argparse.ArgumentParser(
        description="SM100 FA4 q1 decode benchmark for the pack_gqa ping-pong fast path."
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--q1-opt",
        choices=["on", "off", "both"],
        default="both",
        help="Benchmark the SM100 q1 decode optimization enabled, disabled, or both.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Optional JSON produced by this script; prints percent delta versus that baseline.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("benchmark_sm100_decode_q1.py requires an SM100/SM110 CUDA device")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = "cuda"
    torch.manual_seed(1234)

    baseline = _load_baseline(args.baseline_json)
    opt_values = {
        "on": (True,),
        "off": (False,),
        "both": (False, True),
    }[args.q1_opt]
    rows = []
    by_cfg = {}
    for headdim, headdim_v in HDIM_PAIRS:
        for causal in (False, True):
            for seqlen_k, batch in DECODE_CASES:
                for enable_q1_opt in opt_values:
                    ms = _benchmark_case(
                        seqlen_k=seqlen_k,
                        batch=batch,
                        headdim=headdim,
                        headdim_v=headdim_v,
                        causal=causal,
                        dtype=dtype,
                        warmup=args.warmup,
                        rep=args.rep,
                        enable_sm100_decode_q1_opt=enable_q1_opt,
                        device=device,
                    )
                    key = (headdim, headdim_v, causal, seqlen_k, batch, enable_q1_opt)
                    baseline_ms = baseline.get(key)
                    row = {
                        "headdim": headdim,
                        "headdim_v": headdim_v,
                        "causal": causal,
                        "seqlen_k": seqlen_k,
                        "batch": batch,
                        "num_splits": 1,
                        "dtype": args.dtype,
                        "enable_sm100_decode_q1_opt": enable_q1_opt,
                        "ms": ms,
                        "baseline_ms": baseline_ms,
                        "delta": None if baseline_ms is None else (ms / baseline_ms - 1.0),
                    }
                    rows.append(row)
                    by_cfg[(headdim, headdim_v, causal, seqlen_k, batch, enable_q1_opt)] = ms

    for row in rows:
        disabled_ms = by_cfg.get(
            (
                row["headdim"],
                row["headdim_v"],
                row["causal"],
                row["seqlen_k"],
                row["batch"],
                False,
            )
        )
        row["delta_vs_disabled"] = (
            None
            if disabled_ms is None or not row["enable_sm100_decode_q1_opt"]
            else row["ms"] / disabled_ms - 1.0
        )

    has_baseline = any(row["baseline_ms"] is not None for row in rows)
    baseline_hdr = " baseline_ms delta" if has_baseline else ""
    print(
        f"{'hdim':>9} {'causal':>6} {'seqlen_k':>8} {'batch':>5} "
        f"{'q1_opt':>6} {'ms':>9} {'vs_off':>8}{baseline_hdr}"
    )
    for row in rows:
        baseline_ms = row["baseline_ms"]
        suffix = (
            f" {baseline_ms:11.4f} {_format_delta(row['ms'], baseline_ms):>7}"
            if baseline_ms is not None
            else ""
        )
        hdim = (
            str(row["headdim"])
            if row["headdim"] == row["headdim_v"]
            else f"{row['headdim']},{row['headdim_v']}"
        )
        vs_off = _format_delta(row["ms"], by_cfg.get((
            row["headdim"],
            row["headdim_v"],
            row["causal"],
            row["seqlen_k"],
            row["batch"],
            False,
        ))) if row["enable_sm100_decode_q1_opt"] else ""
        print(
            f"{hdim:>9} {str(row['causal']):>6} {row['seqlen_k']:8d} {row['batch']:5d} "
            f"{str(row['enable_sm100_decode_q1_opt']):>6} {row['ms']:9.4f} {vs_off:>8}{suffix}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
