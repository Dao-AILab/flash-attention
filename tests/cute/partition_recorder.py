# Copyright (c) 2026, Tri Dao.
"""Specialization-partition recorder for the FA4 interface refactor.

Records, over a real test-corpus run, how launcher calls partition into
compile-cache entries: how many distinct specializations exist and which call
hits which entry. The dump stores the call sequence relabeled by first
occurrence ("canonical sequence"), so two runs compare equal iff they induce
the same partition — regardless of what the key tuples actually look like.
This is the invariant a key refactor must preserve: fewer entries than the
baseline means collisions (wrong-artifact reuse), more means recompile churn.

The canonical corpus (single process — xdist would interleave orderings).
Only fake-mode-respecting tests qualify: tests that verify kernel-computed
values (test_block_sparsity.py, most of test_mask_mod.py) cannot run with a
stubbed compile; block-sparse key coverage lives in the golden-key matrix
instead.

  FLASH_ATTENTION_FAKE_TENSOR=1 PARTITION_RECORD=1 \
  PARTITION_OUT=tests/cute/data/partition_baseline.json \
  .venv/bin/pytest -q tests/cute/test_flash_attn_fast.py \
      tests/cute/test_flash_attn.py::test_flash_attn_bwd_preallocated_outputs \
      tests/cute/test_flash_attn.py::test_flash_attn_lse_grad

Compare a later run against the baseline with:

  .venv/bin/python tests/cute/partition_recorder.py compare \
      tests/cute/data/partition_baseline.json /tmp/partition_new.json

cute.compile is stubbed (nothing is really compiled; requires fake-tensor
mode), so a corpus run is host-only and fast.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path


class PartitionCache:
    """Dict-shaped cache recording the probe sequence and distinct entries."""

    def __init__(self, name: str):
        self.name = name
        self.store: dict = {}
        self.entry_sha: list[str] = []      # distinct entries, first-seen order
        self.probe_seq: list[int] = []      # per membership probe: entry index
        self._idx: dict[str, int] = {}

    def __contains__(self, key):
        sha = hashlib.sha256(pickle.dumps(key)).hexdigest()
        if sha not in self._idx:
            self._idx[sha] = len(self.entry_sha)
            self.entry_sha.append(sha)
        self.probe_seq.append(self._idx[sha])
        return key in self.store

    def __setitem__(self, key, value):
        self.store[key] = value

    def __getitem__(self, key):
        return self.store[key]

    def clear(self):
        self.store.clear()  # keeps the recorded partition; only entries evict


_CACHES: dict[str, PartitionCache] = {}
_SAVED: dict = {}


def install() -> None:
    """Swap recording caches into every launcher and stub cute.compile."""
    assert os.getenv("FLASH_ATTENTION_FAKE_TENSOR") == "1", (
        "PARTITION_RECORD requires FLASH_ATTENTION_FAKE_TENSOR=1 "
        "(cute.compile is stubbed; kernels must never actually run)"
    )
    assert not os.getenv("PYTEST_XDIST_WORKER"), (
        "PARTITION_RECORD requires a single-process run (no -n): "
        "xdist workers would each record a partial, interleaved partition"
    )
    import flash_attn.cute.interface as fa

    launchers = {
        "fwd": fa._flash_attn_fwd,
        "bwd": fa._flash_attn_bwd,
        "bwd_pre": fa._bwd_preprocess,
        "bwd_post": fa._bwd_postprocess_convert,
        "fwd_combine": fa._flash_attn_fwd_combine,
        "bwd_dsa": fa._flash_attn_bwd_sparse_mla,
        "dq_dqv_gemm": fa._sparse_mla_dq_dqv,
        "dk_gemm": fa._sparse_mla_dk,
    }
    _SAVED["compile"] = fa.cute.compile
    _SAVED["caches"] = {n: fn.compile_cache for n, fn in launchers.items()}
    _SAVED["fa"] = fa
    fa.cute.compile = lambda *a, **k: (lambda *aa, **kk: None)
    for name, fn in launchers.items():
        _CACHES[name] = PartitionCache(name)
        fn.compile_cache = _CACHES[name]


def dump(out_path: str) -> None:
    import torch

    fa = _SAVED["fa"]
    fa.cute.compile = _SAVED["compile"]
    for name, fn in {
        "fwd": fa._flash_attn_fwd, "bwd": fa._flash_attn_bwd,
        "bwd_pre": fa._bwd_preprocess, "bwd_post": fa._bwd_postprocess_convert,
        "fwd_combine": fa._flash_attn_fwd_combine,
        "bwd_dsa": fa._flash_attn_bwd_sparse_mla,
        "dq_dqv_gemm": fa._sparse_mla_dq_dqv, "dk_gemm": fa._sparse_mla_dk,
    }.items():
        fn.compile_cache = _SAVED["caches"][name]
    result = {
        "meta": {
            "commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                text=True, cwd=Path(__file__).parents[2],
            ).stdout.strip(),
            "device": torch.cuda.get_device_name(0),
            "argv": sys.argv,
        },
        "caches": {
            name: {
                "distinct": len(c.entry_sha),
                "num_probes": len(c.probe_seq),
                "canon_sequence": c.probe_seq,   # already first-occurrence order
                "entry_sha": c.entry_sha,        # for debugging only
            }
            for name, c in _CACHES.items() if c.probe_seq
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=1) + "\n")
    print(f"\n[partition_recorder] wrote {out_path}: "
          + ", ".join(f"{n}={d['distinct']}/{d['num_probes']}"
                      for n, d in result["caches"].items()))


def compare(baseline_path: str, new_path: str) -> int:
    """Exit 0 iff the two runs induce identical partitions per cache."""
    base = json.loads(Path(baseline_path).read_text())["caches"]
    new = json.loads(Path(new_path).read_text())["caches"]
    ok = True
    for name in sorted(set(base) | set(new)):
        b, n = base.get(name), new.get(name)
        if b is None or n is None:
            print(f"{name}: present only in {'new' if b is None else 'baseline'}")
            ok = False
            continue
        if b["num_probes"] != n["num_probes"]:
            print(f"{name}: probe count {b['num_probes']} -> {n['num_probes']} "
                  "(corpus drifted; partitions not comparable)")
            ok = False
        elif b["canon_sequence"] != n["canon_sequence"]:
            first = next(i for i, (x, y) in
                         enumerate(zip(b["canon_sequence"], n["canon_sequence"]))
                         if x != y)
            kind = ("collision (fewer entries)" if n["distinct"] < b["distinct"]
                    else "churn (more entries)" if n["distinct"] > b["distinct"]
                    else "relabeled mapping")
            print(f"{name}: partition changed at probe {first}: {kind}; "
                  f"distinct {b['distinct']} -> {n['distinct']}")
            ok = False
        else:
            print(f"{name}: OK ({b['distinct']} entries over {b['num_probes']} probes)")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    sys.exit(f"usage: {sys.argv[0]} compare <baseline.json> <new.json>")
