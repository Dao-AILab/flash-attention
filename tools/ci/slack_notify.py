#!/usr/bin/env python3
"""Post nightly benchmark summary to Slack.

Reads the most recent HISTORY_WINDOW records from benchmark_history.jsonl,
compares today vs yesterday, and flags configs that are below the 7-day
average by more than REGRESSION_THRESHOLD.

Usage:
    SLACK_WEBHOOK_URL=https://hooks.slack.com/... python tools/ci/slack_notify.py
    python tools/ci/slack_notify.py --history benchmark_history.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH = "benchmark-data"
HISTORY_FILE = "benchmark_history.jsonl"

HISTORY_WINDOW = 7          # records to load
REGRESSION_THRESHOLD = 0.02  # alert if today < 7d-avg by more than 2%


# ── History loading ───────────────────────────────────────────────────────────

def fetch_history_from_branch() -> list[dict]:
    try:
        content = subprocess.check_output(
            ["git", "show", f"origin/{BRANCH}:{HISTORY_FILE}"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return []
    return _parse_jsonl(content)


def read_history_from_file(path: Path) -> list[dict]:
    return _parse_jsonl(path.read_text())


def _parse_jsonl(text: str) -> list[dict]:
    records = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ── Analysis ──────────────────────────────────────────────────────────────────

def cfg_key(r: dict) -> tuple:
    return (r["direction"], r.get("hdim", r.get("headdim")),
            r.get("hdim_v", r.get("headdim_v", r.get("hdim", r.get("headdim")))),
            r.get("seqlen_kv", r.get("seqlen")), r.get("seqlen_q", r.get("seqlen_kv", r.get("seqlen"))),
            r["causal"], r.get("group", ""))


def index_results(record: dict) -> dict[tuple, float]:
    """Return {cfg_key: tflops} for a single benchmark record."""
    return {
        cfg_key(r): r["tflops"]
        for r in record.get("results", [])
        if r.get("tflops") is not None
    }


def compute_7d_avg(window: list[dict]) -> dict[tuple, float]:
    """Mean tflops per config across a list of records."""
    from collections import defaultdict
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for rec in window:
        for k, v in index_results(rec).items():
            buckets[k].append(v)
    return {k: statistics.mean(vs) for k, vs in buckets.items()}


def delta_str(now: float, ref: float) -> str:
    pct = (now - ref) / ref * 100
    if abs(pct) < 0.5:
        return "  ~  "
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


# ── Message builder ───────────────────────────────────────────────────────────

GROUP_ORDER = ["mha", "mla_decode", "mla_prefill", "deepseek"]
GROUP_LABELS = {
    "mha":        ":zap: MHA — fwd + bwd",
    "mla_decode": ":robot_face: MLA Decode — fwd",
    "mla_prefill":":robot_face: MLA Prefill — fwd",
    "deepseek":   ":ocean: DeepSeek — fwd",
}


def build_message(records: list[dict]) -> str:
    if not records:
        return "FA4 Nightly: no benchmark history found."

    latest = records[-1]
    arch = latest.get("gpu", {}).get("arch", "")
    same_gpu = [r for r in records if r.get("gpu", {}).get("arch") == arch]

    window = same_gpu[-HISTORY_WINDOW:]
    today = window[-1]
    yesterday = window[-2] if len(window) >= 2 else None
    prior = window[:-1]

    date = today.get("date", "?")
    sha = today.get("sha", "?")
    gpu_name = today.get("gpu", {}).get("name", "?")
    clock_mhz = today.get("clock_mhz")
    clock_str = f" | :lock: {clock_mhz} MHz" if clock_mhz else " | :warning: clocks unknown"

    today_vals = index_results(today)
    yday_vals = index_results(yesterday) if yesterday else {}
    avg_7d = compute_7d_avg(prior) if prior else {}

    date_range = f"{window[0].get('date','?')} → {date}" if len(window) > 1 else date
    n_days = len(window)

    run_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    link = (f" | <{run_url}/{repo}/actions/runs/{run_id}|View run>"
            if repo and run_id else "")

    lines = [f":bar_chart: *FA4 Nightly* — {date} | {gpu_name}{clock_str} | `{sha}`{link}"]
    lines.append(f"_{n_days}-run window: {date_range}_\n")

    has_yday = bool(yday_vals)
    has_avg = bool(avg_7d)
    avg_label = f"vs {n_days-1}d"

    # Bucket keys by group
    from collections import defaultdict
    by_group: dict[str, list] = defaultdict(list)
    for k in today_vals:
        by_group[k[6]].append(k)

    regressions = []

    for group in GROUP_ORDER:
        keys = by_group.get(group)
        if not keys:
            continue

        label = GROUP_LABELS.get(group, group)
        lines.append(f"*{label}*")

        hdr = f"{'dir':<4} {'hdim':>8} {'sq':>6} {'skv':>6} {'csl':>3}  {'TFLOPS':>7}"
        if has_yday:
            hdr += f"  {'vs yday':>7}"
        if has_avg:
            hdr += f"  {avg_label:>7}"
        sep = "─" * len(hdr)
        table_lines = [hdr, sep]

        for k in sorted(keys, key=lambda x: (x[0], x[1], x[2], x[3])):
            direction, hdim, hdim_v, seqlen_kv, seqlen_q, causal, _ = k
            val = today_vals[k]
            hdim_str = str(hdim) if hdim == hdim_v else f"{hdim}-{hdim_v}"
            causal_str = "T" if causal else "F"
            row = f"{direction:<4} {hdim_str:>8} {seqlen_q:>6} {seqlen_kv:>6} {causal_str:>3}  {val:>7.1f}"

            if has_yday and k in yday_vals:
                row += f"  {delta_str(val, yday_vals[k]):>7}"
            elif has_yday:
                row += f"  {'n/a':>7}"

            avg = avg_7d.get(k)
            if has_avg and avg is not None:
                row += f"  {delta_str(val, avg):>7}"
                if (avg - val) / avg > REGRESSION_THRESHOLD:
                    regressions.append((k, val, avg))
            elif has_avg:
                row += f"  {'n/a':>7}"

            table_lines.append(row)

        lines.extend(f"> `{line}`" for line in table_lines)
        lines.append("")

    if regressions:
        lines.append(f":warning: *Regressions (>{REGRESSION_THRESHOLD*100:.0f}% below {n_days-1}d avg):*")
        for k, val, avg in regressions:
            direction, hdim, hdim_v, seqlen_kv, seqlen_q, causal, group = k
            hdim_str = str(hdim) if hdim == hdim_v else f"{hdim}-{hdim_v}"
            drop = (avg - val) / avg * 100
            lines.append(f"  • [{group}] {direction} hdim={hdim_str} sq={seqlen_q} skv={seqlen_kv} causal={causal}: "
                          f"{avg:.1f} → {val:.1f} TFLOPS (*-{drop:.1f}%*)")
    else:
        lines.append(":white_check_mark: No regressions detected.")

    return "\n".join(lines)


# ── Slack posting ─────────────────────────────────────────────────────────────

def post_to_slack(webhook_url: str, message: str) -> None:
    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode()
        if body != "ok":
            print(f"Slack responded: {body}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", type=Path, default=None,
                        help="Local path to benchmark_history.jsonl (default: fetch from branch)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print message instead of posting to Slack")
    args = parser.parse_args()

    if args.history:
        records = read_history_from_file(args.history)
    else:
        subprocess.run(["git", "fetch", "origin", BRANCH], check=False, cwd=REPO_ROOT,
                       capture_output=True)
        records = fetch_history_from_branch()

    # Keep only the last HISTORY_WINDOW records
    records = records[-HISTORY_WINDOW:]

    message = build_message(records)

    if args.dry_run:
        print(message)
        return

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        sys.exit("SLACK_WEBHOOK_URL is not set")
    post_to_slack(webhook_url, message)
    print("Posted to Slack.")


if __name__ == "__main__":
    main()
