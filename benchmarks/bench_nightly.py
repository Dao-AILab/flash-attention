#!/usr/bin/env python3
"""Nightly canonical benchmark — thin wrapper around benchmark_attn.py.

Runs fixed canonical configs (MHA + MLA) for FA4 on SM100, collects
structured results, and writes a single JSON record to --output.

Usage:
    python benchmarks/bench_nightly.py
    python benchmarks/bench_nightly.py --output /tmp/results.json
    python benchmarks/bench_nightly.py --no-lock-clocks  # skip sudo clock lock
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "benchmarks" / "benchmark_attn.py"

# ── Canonical config groups ───────────────────────────────────────────────────
#
# Each entry: (label, extra_cli_args)
# All runs use --backend fa4 --fwd --bwd --causal both unless overridden.
#
CANONICAL_RUNS = [
    # Standard MHA: hdim 64 / 128 / 256, seqlen 4k / 16k, fwd+bwd
    ("mha", [
        "--headdim", "64,128,256",
        "--seqlen", "4096,16384",
        "--fwd", "--bwd", "--causal", "both",
    ]),
    # MLA-absorbed decode: seqlen_q=1, batch=128, seqlen_kv 4k→64k, fwd only
    # paged=False (not yet merged); add --page-size here when ready
    ("mla_decode", [
        "--headdim", "64-512",
        "--nheads", "128", "--nheads-kv", "1",
        "--seqlen-q", "1",
        "--seqlen", "4096,16384,65536",
        "--batch-size", "128",
        "--fwd", "--causal", "true",
    ]),
    # MLA-absorbed prefill: seqlen_q == seqlen_kv, fwd only
    ("mla_prefill", [
        "--headdim", "64-512",
        "--nheads", "128", "--nheads-kv", "1",
        "--seqlen", "4096,16384",
        "--fwd", "--causal", "true",
    ]),
    # DeepSeek shape: hdim=192 hdim_v=128, fwd only (no bwd yet)
    ("deepseek", [
        "--headdim", "192-128",
        "--seqlen", "4096,16384",
        "--fwd", "--causal", "both",
    ]),
]

COMMON_ARGS = ["--backend", "fa4", "--warmup", "5", "--rep", "30"]


# ── Clock locking ─────────────────────────────────────────────────────────────

def _get_gpu_selector() -> Optional[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        entries = [e.strip() for e in visible.split(",") if e.strip()]
        if entries:
            idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
            return entries[idx] if idx < len(entries) else entries[0]
    return str(torch.cuda.current_device()) if torch.cuda.is_available() else None


def _nvidia_smi_cmd(*args) -> list[str]:
    prefix: list[str] = [] if os.geteuid() == 0 else ["sudo"]
    cmd = prefix + ["nvidia-smi"]
    sel = _get_gpu_selector()
    if sel is not None:
        cmd += ["-i", sel]
    return cmd + list(args)


def _query_clocks() -> tuple[Optional[str], Optional[str]]:
    try:
        r = subprocess.run(
            _nvidia_smi_cmd("--query-gpu=clocks.current.graphics,clocks.max.graphics",
                            "--format=csv,noheader,nounits"),
            capture_output=True, text=True,
        )
    except OSError:
        return None, None
    if r.returncode != 0:
        return None, None
    lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None, None
    fields = [f.strip() for f in lines[0].split(",")]
    return (fields[0], fields[1]) if len(fields) >= 2 else (None, None)


def setup_clocks(lock: bool) -> None:
    cur, max_clk = _query_clocks()
    if cur is None:
        print("WARNING: could not query GPU clocks")
        return
    if not lock:
        if cur != max_clk:
            print(f"WARNING: clocks not locked ({cur}/{max_clk} MHz) — results may vary")
        return
    if cur == max_clk:
        print(f"GPU clocks already at max ({max_clk} MHz).")
        return
    try:
        r = subprocess.run(_nvidia_smi_cmd("--lock-gpu-clocks", max_clk),
                           capture_output=True, text=True)
    except OSError as e:
        print(f"WARNING: could not lock clocks ({e})")
        return
    if r.returncode == 0:
        print(f"Locked GPU clocks to {max_clk} MHz.")
        atexit.register(lambda: subprocess.run(
            _nvidia_smi_cmd("--reset-gpu-clocks"), capture_output=True))
    else:
        print(f"WARNING: clock lock failed ({r.stderr.strip()})")


# ── GPU / git metadata ────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return {"name": torch.cuda.get_device_name(), "sm": sm, "arch": f"sm{sm}",
            "count": torch.cuda.device_count()}


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ── Run one canonical group ───────────────────────────────────────────────────

def run_group(label: str, extra_args: list[str]) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        cmd = [sys.executable, str(BENCH_SCRIPT),
               *COMMON_ARGS, *extra_args,
               "--json-output", tmp]
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        with open(tmp) as f:
            results = json.load(f)
        for r in results:
            r["group"] = label
        return results
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="bench_results.json",
                        help="Output JSON file (default: bench_results.json)")
    parser.add_argument("--lock-clocks", action=argparse.BooleanOptionalAction, default=True,
                        help="Lock GPU clocks before benchmarking (requires sudo, default: on)")
    parser.add_argument("--groups", default=None,
                        help="Comma-separated subset of groups to run (default: all). "
                             f"Choices: {','.join(l for l,_ in CANONICAL_RUNS)}")
    args = parser.parse_args()

    gpu = get_gpu_info()
    print(f"GPU: {gpu['name']} ({gpu['arch']})")
    setup_clocks(args.lock_clocks)

    selected = set(args.groups.split(",")) if args.groups else None
    all_results = []
    for label, extra_args in CANONICAL_RUNS:
        if selected and label not in selected:
            continue
        results = run_group(label, extra_args)
        all_results.extend(results)
        print(f"  → {len(results)} results collected")

    record = {
        "date": datetime.date.today().isoformat(),
        "sha": get_git_sha(),
        "gpu": gpu,
        "hostname": socket.gethostname(),
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(record, f, separators=(",", ":"))
        f.write("\n")
    print(f"\nTotal {len(all_results)} results → {args.output}")


if __name__ == "__main__":
    main()
