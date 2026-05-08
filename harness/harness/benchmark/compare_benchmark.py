#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROW_RE = re.compile(
    r"^(?P<mask>non-causal|causal)\s+/\s+seqlen=(?P<seqlen>\d+)\s+"
    r"(?P<batch>\d+)\s+(?P<fa_ms>\d+(?:\.\d+)?)\s+"
    r"(?P<tflops>\d+(?:\.\d+)?)(?:\(\d+(?:\.\d+)?%\))?"
)


def parse_logs(directory: Path) -> dict[tuple[str, str, int], list[float]]:
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    if not directory.exists():
        return values
    for path in sorted(directory.glob("run_*.log")):
        direction = "unknown"
        for line in path.read_text(errors="replace").splitlines():
            if "[Forward]" in line:
                direction = "Forward"
            elif "[Backward]" in line:
                direction = "Backward"
            match = ROW_RE.match(line)
            if not match:
                continue
            key = (direction, match.group("mask"), int(match.group("seqlen")))
            values[key].append(float(match.group("tflops")))
    return values


def median_map(values: dict[tuple[str, str, int], list[float]]) -> dict[tuple[str, str, int], float]:
    return {key: statistics.median(v) for key, v in values.items() if v}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--regression-threshold", type=float, default=0.03)
    args = parser.parse_args()

    current = median_map(parse_logs(args.current))
    previous = median_map(parse_logs(args.previous))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Report",
        "",
        f"Current: `{args.current}`",
        f"Previous: `{args.previous}`",
        f"Regression threshold: `{args.regression_threshold:.1%}`",
        "",
    ]

    if not current:
        lines += ["No current benchmark rows parsed.", ""]
        args.report.write_text("\n".join(lines))
        return 2

    if not previous:
        lines += ["No previous benchmark directory found. Current run becomes the baseline for the next comparison.", ""]
        lines += ["| Direction | Mask | Seqlen | Current Median TFLOPS |", "| --------- | ---- | ------ | --------------------- |"]
        for key in sorted(current):
            direction, mask, seqlen = key
            lines.append(f"| {direction} | {mask} | {seqlen} | {current[key]:.1f} |")
        args.report.write_text("\n".join(lines) + "\n")
        print(f"[benchmark] wrote {args.report}")
        return 0

    regressions: list[tuple[tuple[str, str, int], float, float, float]] = []
    lines += [
        "| Direction | Mask | Seqlen | Previous Median TFLOPS | Current Median TFLOPS | Delta | Status |",
        "| --------- | ---- | ------ | ---------------------- | --------------------- | ----- | ------ |",
    ]
    for key in sorted(set(previous) | set(current)):
        prev = previous.get(key)
        cur = current.get(key)
        direction, mask, seqlen = key
        if prev is None:
            lines.append(f"| {direction} | {mask} | {seqlen} | n/a | {cur:.1f} | n/a | new |")
            continue
        if cur is None:
            lines.append(f"| {direction} | {mask} | {seqlen} | {prev:.1f} | n/a | n/a | missing |")
            regressions.append((key, prev, 0.0, -1.0))
            continue
        delta = (cur - prev) / prev if prev else 0.0
        status = "REGRESSION" if delta < -args.regression_threshold else "ok"
        if status == "REGRESSION":
            regressions.append((key, prev, cur, delta))
        lines.append(f"| {direction} | {mask} | {seqlen} | {prev:.1f} | {cur:.1f} | {delta:.1%} | {status} |")

    if regressions:
        lines += [
            "",
            "## Regression Action",
            "",
            "Systemic regression detected from repeated benchmark medians.",
            "Export SASS for before/after kernels or run equivalent experiment analysis before modifying further.",
            "Do not submit or lock this round until performance is recovered.",
            "",
        ]
    else:
        lines += ["", "No systemic regression detected.", ""]

    args.report.write_text("\n".join(lines) + "\n")
    print(f"[benchmark] wrote {args.report}")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
