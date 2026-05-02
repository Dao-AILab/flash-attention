#!/usr/bin/env python3
"""Append a benchmark result JSON file to the benchmark-data branch.

Creates the branch (orphan) if it doesn't exist yet.

Usage:
    python tools/ci/push_results.py --result bench_results.json
    python tools/ci/push_results.py --result bench_results.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


BRANCH = "benchmark-data"
HISTORY_FILE = "benchmark_history.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("check", True)
    kwargs.setdefault("cwd", REPO_ROOT)
    return subprocess.run(cmd, **kwargs)


def run_out(cmd: list[str], **kwargs) -> str:
    kwargs.setdefault("cwd", REPO_ROOT)
    return subprocess.check_output(cmd, **kwargs).decode().strip()


def branch_exists_on_remote() -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", BRANCH],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return bool(result.stdout.strip())


def setup_worktree(worktree_path: Path) -> None:
    """Check out benchmark-data into a temp worktree (create orphan if needed)."""
    if branch_exists_on_remote():
        run(["git", "fetch", "origin", f"{BRANCH}:{BRANCH}"], check=False)
        run(["git", "worktree", "add", str(worktree_path), BRANCH])
    else:
        # Orphan branch — no history, no files
        run(["git", "worktree", "add", "--orphan", "-b", BRANCH, str(worktree_path)])


def push_result(result_path: Path, dry_run: bool) -> None:
    result_text = result_path.read_text().strip()
    # Validate it's valid JSON
    record = json.loads(result_text)
    date = record.get("date", "unknown")
    sha = record.get("sha", "unknown")
    gpu_name = record.get("gpu", {}).get("name", "unknown")
    arch = record.get("gpu", {}).get("arch", "unknown")

    with tempfile.TemporaryDirectory(prefix="fa4-bench-") as tmpdir:
        worktree = Path(tmpdir) / "bench-data"
        setup_worktree(worktree)
        try:
            history = worktree / HISTORY_FILE
            with open(history, "a") as f:
                f.write(result_text + "\n")

            subprocess.run(["git", "add", HISTORY_FILE], check=True, cwd=worktree)
            commit_msg = f"nightly: {date} {arch} {sha[:7]}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg,
                 "--author", "FA4 Nightly <nightly@together.ai>"],
                check=True, cwd=worktree,
            )

            n_results = len(record.get("results", []))
            print(f"Committed: {commit_msg} ({n_results} configs, GPU: {gpu_name})")

            if dry_run:
                print("[dry-run] skipping push")
            else:
                subprocess.run(["git", "push", "origin", BRANCH], check=True, cwd=worktree)
                print(f"Pushed to origin/{BRANCH}")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                           check=False, cwd=REPO_ROOT)
            # Clean up local branch ref if we created it as orphan
            if not branch_exists_on_remote() and not dry_run:
                subprocess.run(["git", "branch", "-D", BRANCH],
                                check=False, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True, help="Path to bench_results.json")
    parser.add_argument("--dry-run", action="store_true", help="Commit but don't push")
    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        sys.exit(f"Result file not found: {result_path}")

    push_result(result_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
