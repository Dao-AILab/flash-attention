#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


FAIL_RE = re.compile(
    r"(^|\n)(FAILED\s+|FAIL\s+)|=+\s+[^\n]*\b([1-9]\d*)\s+failed\b",
    re.MULTILINE,
)
CASE_RE = re.compile(r"^(FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def run(cmd: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        bufsize=1,
    )


def kill_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(2)
    except ProcessLookupError:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def gpu_util() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        return None
    vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
    return max(vals) if vals else None


def read_new(path: Path, offsets: dict[Path, int]) -> str:
    if not path.exists():
        return ""
    offset = offsets.get(path, 0)
    with path.open("r", errors="replace") as f:
        f.seek(offset)
        data = f.read()
        offsets[path] = f.tell()
    return data


def extract_cases(text: str) -> list[str]:
    return [m.group(2) for m in CASE_RE.finditer(text)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--hang-seconds", type=int, default=45)
    parser.add_argument("--sample-seconds", type=int, default=5)
    parser.add_argument("--gpu-100-samples", type=int, default=3)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    harness_root = script_dir.parents[1]
    repo = harness_root.parent
    test_log_dir = harness_root / "logs" / "test"
    test_runner_log_dir = harness_root / "logs" / "test"
    test_log_dir.mkdir(parents=True, exist_ok=True)
    monitor_log = test_log_dir / "monitor.log"
    failure_cases = test_log_dir / "failure_cases.txt"
    hang_report = test_log_dir / "hang_report.txt"

    cmd = ["bash", "harness/harness/test/run_hd256_ut.sh"]
    if args.preflight_only:
        cmd.append("--preflight-only")

    with monitor_log.open("w") as log:
        def note(msg: str) -> None:
            line = f"[monitor] {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        note(f"repo={repo}")
        note(f"cmd={' '.join(cmd)}")
        test_runner_log_dir.mkdir(parents=True, exist_ok=True)
        for old_log in test_runner_log_dir.glob("*.log"):
            if old_log.name != "monitor.log":
                old_log.unlink()
        proc = run(cmd)
        start = time.monotonic()
        offsets: dict[Path, int] = {}
        gpu_100_streak = 0
        collected = ""

        while True:
            if proc.stdout is not None and select.select([proc.stdout], [], [], 0)[0]:
                line = proc.stdout.readline()
                if line:
                    print(line, end="")
                    log.write(line)
                    log.flush()
                    collected += line
                    if FAIL_RE.search(line):
                        note("failure detected in command output; killing UT process group")
                        kill_process_group(proc)
                        return 1

            for path in sorted(test_runner_log_dir.glob("*.log")):
                if path == monitor_log:
                    continue
                chunk = read_new(path, offsets)
                if chunk:
                    collected += "\n" + chunk
                    if FAIL_RE.search(chunk):
                        cases = extract_cases(collected)
                        failure_cases.write_text(
                            "\n".join(cases)
                            + ("\n" if cases else "")
                            + "\nReproduce with:\n"
                            + "\n".join(f"python3 -m pytest -v -s --tb=long {c}" for c in cases)
                            + "\n",
                        )
                        note(f"failure detected in {path}; killing UT process group")
                        kill_process_group(proc)
                        return 1

            if args.preflight_only and proc.poll() is not None:
                return proc.returncode or 0

            elapsed = time.monotonic() - start
            util = gpu_util()
            if util == 100:
                gpu_100_streak += 1
            else:
                gpu_100_streak = 0

            if (
                elapsed >= args.hang_seconds
                and gpu_100_streak >= args.gpu_100_samples
                and proc.poll() is None
            ):
                msg = (
                    f"hang detected: elapsed={elapsed:.1f}s, "
                    f"gpu_util=100 for {gpu_100_streak} samples"
                )
                hang_report.write_text(
                    msg
                    + "\nKilled broad UT run. Reproduce smallest hanging case, then use:\n"
                    + "bash harness/harness/hang_detect_fix/cuda_gdb_snapshot.sh <pid>\n"
                )
                note(msg + "; killing UT process group")
                kill_process_group(proc)
                return 2

            if proc.poll() is not None:
                rest = proc.stdout.read() if proc.stdout is not None else ""
                if rest:
                    print(rest, end="")
                    log.write(rest)
                    collected += rest
                if proc.returncode != 0 and FAIL_RE.search(collected):
                    cases = extract_cases(collected)
                    failure_cases.write_text("\n".join(cases) + ("\n" if cases else ""))
                return proc.returncode or 0

            time.sleep(args.sample_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
