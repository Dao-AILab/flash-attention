import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

BENCHMARK = Path(__file__).with_name("fwd_config_bench.py")
SPEC = importlib.util.spec_from_file_location("fwd_config_bench", BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def test_campaign_is_data_only_and_names_every_grid_axis(tmp_path):
    campaign = {
        "experiments": [
            {
                "name": "axes",
                "baseline": {"use_clc_scheduler": False},
                "candidate": {"use_clc_scheduler": True},
                "correctness": {
                    "mode": "dense",
                    "q_heads": 8,
                    "kv_heads": 8,
                    "head_dim": 64,
                    "batch": 1,
                    "seqlen_q": 8,
                    "seqlen_k": 16,
                },
                "grids": [
                    {
                        "phase": "discovery",
                        "mode": "dense",
                        "head_pairs": [[8, 8]],
                        "head_dims": [64],
                        "batches": [1],
                        "seqlen_pairs": [[16, 32]],
                        "causal": [False, True],
                        "num_splits": [1, 2],
                    }
                ],
                "case_table": {
                    "mode": "dense",
                    "columns": [
                        "phase",
                        "q_heads",
                        "kv_heads",
                        "head_dim",
                        "batch",
                        "seqlen_q",
                        "seqlen_k",
                    ],
                    "rows": [["holdout", 8, 8, 64, 1, 17, 33]],
                },
            }
        ]
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(campaign))

    _, experiments, cases, correctness = bench.load_campaign(path)

    assert len(experiments) == len(correctness) == 1
    assert len(cases) == len({case.name for case in cases}) == 5
    assert all("_c" in case.name and "_s" in case.name for case in cases)


def test_summary_emits_one_geomean_and_timing_interval_per_policy():
    experiments = {"policy": {"baseline": {"flag": False}, "candidate": {"flag": True}}}
    rows = [
        {
            "experiment": "policy",
            "phase": phase,
            "speedup": speedup,
            "baseline_median_us": baseline,
            "candidate_median_us": candidate,
            "baseline_round_medians_us": [baseline, baseline],
            "candidate_round_medians_us": [candidate, candidate],
        }
        for phase, speedup, baseline, candidate in (
            ("boundary", 2.0, 20.0, 10.0),
            ("holdout", 0.5, 10.0, 20.0),
        )
    ]

    summary = bench.summarize(rows, experiments)

    assert len(summary) == 1
    assert summary[0]["cases"] == 2
    assert summary[0]["phases"] == ["boundary", "holdout"]
    assert summary[0]["geomean_speedup"] == pytest.approx(1.0)
    assert summary[0]["time_weighted_speedup"] == pytest.approx(1.0)
    assert summary[0]["minimum_speedup"] == 0.5
    assert summary[0]["maximum_speedup"] == 2.0
    assert summary[0]["geomean_ci95"] == pytest.approx([1.0, 1.0])
