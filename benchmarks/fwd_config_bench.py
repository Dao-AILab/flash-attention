"""A/B timing of explicit FA4 forward configs.

Apply baseline/candidate overrides to each case's default config, check agreement,
and time interleaved CUDA-graph rounds. Speedup = baseline / candidate median time
(higher favors the candidate).

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
import statistics
from itertools import accumulate
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
        return {
            "q": rand(batch, seqlen_q, q_heads, head_dim),
            "k": rand(batch, seqlen_k, kv_heads, head_dim),
            "v": rand(batch, seqlen_k, kv_heads, head_dim),
            "causal": causal,
        }
    assert len(seqlen_q) == len(seqlen_k) == batch

    def cu_seqlens(lengths):
        return torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)

    return {
        "q": rand(sum(seqlen_q), q_heads, head_dim),
        "k": rand(sum(seqlen_k), kv_heads, head_dim),
        "v": rand(sum(seqlen_k), kv_heads, head_dim),
        "causal": causal,
        "cu_seqlens_q": cu_seqlens(seqlen_q),
        "cu_seqlens_k": cu_seqlens(seqlen_k),
        "max_seqlen_q": max(seqlen_q),
        "max_seqlen_k": max(seqlen_k),
    }


def default_config(kwargs):
    """Capture config=None selection and the interface metadata it selected from."""
    selected = []

    def record(inputs):
        selected.append((fwd_config.select_fwd_config(inputs), inputs))
        return selected[-1][0]

    with mock.patch.object(interface, "select_fwd_config", record):
        interface._flash_attn_fwd(**kwargs)
    return selected[0]


def run_case(experiment, kwargs, rounds):
    selected, inputs = default_config(kwargs)
    # Overrides re-derive registers and the SMEM layout unless they set them.
    arms = [
        fwd_config.derive_fwd_config(selected, inputs, **experiment[name])
        for name in ("baseline", "candidate")
    ]
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
        geomean = statistics.geometric_mean(speedups)
        print(
            f"{experiment['name']}: n={len(speedups)} geomean={geomean:.3f}x "
            f"min={min(speedups):.3f}x max={max(speedups):.3f}x "
            f"losses={sum(s < 1 for s in speedups)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
