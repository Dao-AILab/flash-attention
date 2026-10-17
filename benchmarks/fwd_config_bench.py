#!/usr/bin/env python3
"""A/B-time explicit FA4 forward configs.

Each experiment applies `baseline` and `candidate` field overrides to the config that
`config=None` selects for every case, checks that both arms agree, and times them
with interleaved CUDA-graph rounds. The reported speedup is baseline / candidate
median time (higher favors the candidate).

  python benchmarks/fwd_config_bench.py campaign.yaml [--experiment NAME] [--rounds N]

campaign.yaml:
  experiments:
    - name: d64_direct_o
      baseline: {use_tma_o: true}
      candidate: {use_tma_o: false}
      varlen: false
      cases:  # [batch, q_heads, kv_heads, head_dim, seqlen_q, seqlen_k, causal, label?]
        - [4, 16, 16, 64, 8192, 8192, false]

For varlen experiments, seqlen_q and seqlen_k are lists of per-sequence lengths.
"""

import argparse
import dataclasses
import math
import statistics
from unittest import mock

import torch
from triton.testing import do_bench_cudagraph

try:
    import yaml
except ImportError as e:
    raise SystemExit("Missing pyyaml. Install it with: uv pip install pyyaml") from e

from flash_attn.cute import config as fwd_config
from flash_attn.cute import interface


def make_kwargs(batch, q_heads, kv_heads, head_dim, seqlen_q, seqlen_k, causal, *, varlen):
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    if not varlen:
        return dict(
            q=rand(batch, seqlen_q, q_heads, head_dim),
            k=rand(batch, seqlen_k, kv_heads, head_dim),
            v=rand(batch, seqlen_k, kv_heads, head_dim),
            causal=causal,
        )
    assert len(seqlen_q) == len(seqlen_k) == batch

    def cu_seqlens(lengths):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, device="cuda", dtype=torch.int32)

    return dict(
        q=rand(sum(seqlen_q), q_heads, head_dim),
        k=rand(sum(seqlen_k), kv_heads, head_dim),
        v=rand(sum(seqlen_k), kv_heads, head_dim),
        causal=causal,
        cu_seqlens_q=cu_seqlens(seqlen_q),
        cu_seqlens_k=cu_seqlens(seqlen_k),
        max_seqlen_q=max(seqlen_q),
        max_seqlen_k=max(seqlen_k),
    )


def default_config(kwargs):
    """Return the config that config=None selects from the interface's own inputs."""
    selected = []

    def record(inputs):
        selected.append(fwd_config.select_fwd_config(inputs))
        return selected[-1]

    with mock.patch.object(interface, "select_fwd_config", record):
        interface._flash_attn_fwd(**kwargs)
    return selected[0]


def run_case(experiment, kwargs, rounds):
    selected = default_config(kwargs)
    arms = [dataclasses.replace(selected, **experiment[name]) for name in ("baseline", "candidate")]
    if arms[0] == arms[1]:
        raise ValueError(f"{experiment['name']}: baseline and candidate resolve to {arms[0]}")
    fns = [lambda config=config: interface._flash_attn_fwd(**kwargs, config=config)[0] for config in arms]
    torch.testing.assert_close(fns[1](), fns[0](), atol=2e-2, rtol=2e-2)
    times = [[], []]
    for round_idx in range(rounds):
        for arm in (round_idx % 2, 1 - round_idx % 2):
            times[arm].append(do_bench_cudagraph(fns[arm], rep=20, return_mode="median"))
    return statistics.median(times[0]) / statistics.median(times[1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("campaign")
    parser.add_argument("--experiment")
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    with open(args.campaign) as f:
        experiments = yaml.safe_load(f)["experiments"]
    for experiment in experiments:
        if args.experiment not in (None, experiment["name"]):
            continue
        speedups = []
        for case in experiment["cases"]:
            torch.manual_seed(0)
            kwargs = make_kwargs(*case[:7], varlen=experiment.get("varlen", False))
            speedup = run_case(experiment, kwargs, args.rounds)
            speedups.append(speedup)
            print(f"{experiment['name']} {case}: {speedup:.3f}x", flush=True)
        geomean = math.exp(statistics.mean(map(math.log, speedups)))
        print(
            f"{experiment['name']}: n={len(speedups)} geomean={geomean:.3f}x "
            f"min={min(speedups):.3f}x max={max(speedups):.3f}x "
            f"losses={sum(s < 1 for s in speedups)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
