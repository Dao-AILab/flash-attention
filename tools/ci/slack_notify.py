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
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH = "benchmark-data"
HISTORY_FILE = "benchmark_history.jsonl"

HISTORY_WINDOW = 7          # records to load
REGRESSION_THRESHOLD = 0.02  # alert if today < 7d-avg by more than 2%

GROUP_ORDER = ["mha", "mla_decode", "mla_prefill", "deepseek"]
GROUP_LABELS = {
    "mha":         ":zap: MHA — fwd + bwd",
    "mla_decode":  ":robot_face: MLA Decode — fwd",
    "mla_prefill": ":robot_face: MLA Prefill — fwd",
    "deepseek":    ":ocean: DeepSeek — fwd",
}


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
    return {
        cfg_key(r): r["tflops"]
        for r in record.get("results", [])
        if r.get("tflops") is not None
    }


def compute_7d_avg(window: list[dict]) -> dict[tuple, float]:
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


# ── Report builder ────────────────────────────────────────────────────────────

def _prepare_report(records: list[dict]) -> dict | None:
    """Compute everything needed to render the message. Returns None if no data."""
    if not records:
        return None

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
    clock_str = f":lock: {clock_mhz} MHz" if clock_mhz else ":warning: clocks unknown"

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

    has_yday = bool(yday_vals)
    has_avg = bool(avg_7d)

    by_group: dict[str, list] = defaultdict(list)
    for k in today_vals:
        by_group[k[6]].append(k)

    # Build per-group table text and collect regressions
    groups = []
    regressions = []
    for group in GROUP_ORDER:
        keys = by_group.get(group)
        if not keys:
            continue
        label = GROUP_LABELS.get(group, group)
        hdr = f"{'op':<4} {'hdim':>8} {'seqlen_q':>8} {'seqlen_kv':>9} {'causal':>6}  {'TFLOPS':>7}"
        if has_yday:
            hdr += f"  {'Δ yday':>7}"
        if has_avg:
            hdr += f"  {f'Δ {n_days-1}d-avg':>9}"
        table_lines = [hdr, "─" * len(hdr)]
        for k in sorted(keys, key=lambda x: (x[0], x[1], x[2], x[3])):
            direction, hdim, hdim_v, seqlen_kv, seqlen_q, causal, _ = k
            val = today_vals[k]
            hdim_str = str(hdim) if hdim == hdim_v else f"{hdim}-{hdim_v}"
            row = f"{direction:<4} {hdim_str:>8} {seqlen_q:>8} {seqlen_kv:>9} {str(causal):>6}  {val:>7.1f}"
            if has_yday and k in yday_vals:
                row += f"  {delta_str(val, yday_vals[k]):>7}"
            elif has_yday:
                row += f"  {'n/a':>7}"
            avg = avg_7d.get(k)
            if has_avg and avg is not None:
                row += f"  {delta_str(val, avg):>9}"
                if (avg - val) / avg > REGRESSION_THRESHOLD:
                    regressions.append((k, val, avg))
            elif has_avg:
                row += f"  {'n/a':>9}"
            table_lines.append(row)
        groups.append((label, "\n".join(table_lines)))

    if regressions:
        reg_lines = [f":warning: *Regressions (>{REGRESSION_THRESHOLD*100:.0f}% below {n_days-1}d avg):*"]
        for k, val, avg in regressions:
            direction, hdim, hdim_v, seqlen_kv, seqlen_q, causal, group = k
            hdim_str = str(hdim) if hdim == hdim_v else f"{hdim}-{hdim_v}"
            drop = (avg - val) / avg * 100
            reg_lines.append(
                f"  • [{group}] {direction} hdim={hdim_str} seqlen_q={seqlen_q} "
                f"seqlen_kv={seqlen_kv} causal={causal}: "
                f"{avg:.1f} → {val:.1f} TFLOPS (*-{drop:.1f}%*)"
            )
        regression_text = "\n".join(reg_lines)
    else:
        regression_text = ":white_check_mark: No regressions detected."

    return dict(
        header=f":bar_chart: *FA4 Nightly* — {date} | {gpu_name} | {clock_str} | `{sha}`{link}",
        window_line=f"_{n_days}-run window: {date_range}_",
        groups=groups,
        regression_text=regression_text,
        date=date, gpu_name=gpu_name,
    )


# ── Slack payload (Block Kit) ─────────────────────────────────────────────────

def build_payload(records: list[dict]) -> dict:
    """Build Slack Block Kit payload. Uses rich_text_preformatted for compact monospace tables."""
    report = _prepare_report(records)
    if report is None:
        return {"text": "FA4 Nightly: no benchmark history found."}

    blocks = []
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": report["header"] + "\n" + report["window_line"]},
    })

    for label, table_text in report["groups"]:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{label}*"},
        })
        blocks.append({
            "type": "rich_text",
            "elements": [{
                "type": "rich_text_preformatted",
                "elements": [{"type": "text", "text": table_text}],
            }],
        })

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": report["regression_text"]},
    })

    return {
        "text": f"FA4 Nightly {report['date']} | {report['gpu_name']}",  # notification fallback
        "blocks": blocks,
    }


# ── Dry-run text (terminal preview) ──────────────────────────────────────────

def build_message(records: list[dict]) -> str:
    """Plain-text version for --dry-run."""
    report = _prepare_report(records)
    if report is None:
        return "FA4 Nightly: no benchmark history found."

    lines = [report["header"], report["window_line"], ""]
    for label, table_text in report["groups"]:
        lines.append(f"*{label}*")
        lines.append(table_text)
        lines.append("")
    lines.append(report["regression_text"])
    return "\n".join(lines)


# ── Slack posting ─────────────────────────────────────────────────────────────

def post_to_slack(webhook_url: str, payload: dict) -> None:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook_url,
        data=data,
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

    records = records[-HISTORY_WINDOW:]

    if args.dry_run:
        print(build_message(records))
        return

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        sys.exit("SLACK_WEBHOOK_URL is not set")
    post_to_slack(webhook_url, build_payload(records))
    print("Posted to Slack.")


if __name__ == "__main__":
    main()
