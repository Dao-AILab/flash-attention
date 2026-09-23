#!/usr/bin/env python3
"""Resolve the next FA4 beta tag and optionally create + push it.

Usage:
    python bump_beta_tag.py              # dry-run by default
    python bump_beta_tag.py --push       # create and push the tag
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

TAG_PATTERN = re.compile(r"^(fa4-v.+\.beta)(\d+)$")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def get_beta_tags() -> list[tuple[str, int]]:
    raw = git("tag", "-l", "fa4-v*.beta*")
    if not raw:
        return []
    tags = []
    for line in raw.splitlines():
        m = TAG_PATTERN.match(line.strip())
        if m:
            tags.append((line.strip(), int(m.group(2))))
    return sorted(tags, key=lambda t: t[1])


def tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", tag], capture_output=True, text=True
    )
    return result.returncode == 0


def set_github_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Create and push the tag (default: dry-run)")
    args = parser.parse_args()

    tags = get_beta_tags()
    if not tags:
        print("::error::No existing fa4-v*.beta* tags found", file=sys.stderr)
        sys.exit(1)

    latest_tag, latest_num = tags[-1]
    next_num = latest_num + 1
    prefix = TAG_PATTERN.match(latest_tag)
    if prefix is None:
        print(f"::error::Latest tag {latest_tag!r} no longer matches pattern", file=sys.stderr)
        sys.exit(1)
    next_tag = f"{prefix.group(1)}{next_num}"

    already_exists = tag_exists(next_tag)

    if already_exists:
        print(f"Tag {next_tag} already exists, reusing it")
    else:
        print(f"Bumping: {latest_tag} -> {next_tag}")

    set_github_output("next_tag", next_tag)

    if args.push and not already_exists:
        try:
            git("tag", next_tag)
            git("push", "origin", next_tag)
        except subprocess.CalledProcessError:
            if tag_exists(next_tag):
                print(f"Tag {next_tag} was created by a concurrent run, reusing it")
            else:
                raise
        else:
            print(f"Pushed {next_tag}")
    elif not args.push:
        print(f"Dry-run: would create and push {next_tag}")


if __name__ == "__main__":
    main()
