# Copyright (c) 2025 Aliyun Inc. All Rights Reserved.
"""Call counters for SM100 head_dim=256 2CTA kernels (diagnostics / regression tests)."""

_fwd = 0
_bwd = 0


def reset() -> None:
    global _fwd, _bwd
    _fwd = 0
    _bwd = 0


def bump_fwd() -> None:
    global _fwd
    _fwd += 1


def bump_bwd() -> None:
    global _bwd
    _bwd += 1


def counts() -> dict[str, int]:
    return {"fwd": _fwd, "bwd": _bwd}
