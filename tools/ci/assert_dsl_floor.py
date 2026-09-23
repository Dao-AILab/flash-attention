#!/usr/bin/env python3
"""Fail loudly if the CI image's deps are below the flash_attn/cute/pyproject.toml floors.

Runs inside the SIF before tests. The FA4 install in run_fa4_ci.py uses --no-deps (to keep the
SIF's torch/cudnn), so pyproject floors are not enforced at install time. A SIF baked before a
floor bump therefore keeps a stale dep — e.g. nvidia-cutlass-dsl 4.4.2, which can't convert the
AuxData JIT arg and dies with a cryptic DSLRuntimeError deep in SM100 kernel launch. This check
turns that into an actionable "rebake the image" message up front.

Reads the floor from pyproject so there is no hardcoded version here to drift out of sync.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 (pyproject declares requires-python >=3.10)
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        sys.exit(
            "ERROR: assert_dsl_floor.py needs a TOML parser — use Python 3.11+ (stdlib tomllib) "
            "or `pip install tomli` on 3.10."
        )

# Deps whose floor a stale SIF is known to silently violate. Other pyproject deps (torch, einops…)
# are baked to match the image and not version-sensitive in the same way, so we don't gate on them.
CHECKED = ("nvidia-cutlass-dsl", "quack-kernels", "apache-tvm-ffi")


def main(pyproject_path: str) -> int:
    with open(pyproject_path, "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    reqs = {r.name: r for r in (Requirement(d) for d in deps) if r.name in CHECKED}

    failures: list[str] = []
    oks: list[str] = []
    for name in CHECKED:
        req = reqs.get(name)
        if req is None:
            continue  # not a hard dep in this pyproject — nothing to enforce
        try:
            installed = version(name)
        except PackageNotFoundError:
            failures.append(f"{name}: not installed (floor {req.specifier})")
            continue
        if req.specifier.contains(Version(installed), prereleases=True):
            oks.append(f"{name}={installed}")
        else:
            failures.append(f"{name}: installed {installed} does not satisfy floor {req.specifier}")

    if failures:
        print("ERROR: CI image deps are below the flash_attn/cute/pyproject.toml floor:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nThe SIF was likely baked before a floor bump. Rebake the image "
            "(tools/ci/docker/build.sh + tag_and_push.sh) and update the digest in "
            ".github/workflows/ci.yml.",
            file=sys.stderr,
        )
        return 1

    print("DSL floor check OK: " + ", ".join(oks))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "flash_attn/cute/pyproject.toml"))
