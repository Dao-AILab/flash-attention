from collections import Counter
from pathlib import Path

from flex_flash_block_sparse import generate_flex_block_sparse_cases
from fwd_heuristics_bench import aggregate_bucket_results
from realistic_workloads import (
    estimate_case_bytes,
    generate_realistic_workloads,
    load_workload_spec,
)

CONFIG = Path(__file__).parent / "configs" / "fwd_realistic_workloads.yaml"
GB300_MEMORY_BYTES = 284_208 * 1024**2


def aggregate_test_row(name, production, common, extras):
    """Build one synthetic benchmark row for aggregation tests."""
    config_results = [
        {"name": "production_default", "config": production, "median_us": 100.0},
        {"name": "common", "config": common, "median_us": 90.0},
        *[
            {"name": extra_name, "config": config, "median_us": median_us}
            for extra_name, config, median_us in extras
        ],
    ]
    return {
        "name": name,
        "bucket_name": "shared-bucket",
        "phase": "discovery",
        "production_config": production,
        "winner_config": min(config_results, key=lambda item: item["median_us"])[
            "config"
        ],
        "config_results": config_results,
    }


def test_realistic_workload_spec_has_sourced_profiles_and_holdouts():
    spec = load_workload_spec(CONFIG)

    assert len(spec["model_profiles"]) == 15
    assert len(spec["scenario_templates"]["holdout"]) == 25
    assert all(
        profile["source"].startswith("https://") for profile in spec["model_profiles"]
    )


def test_realistic_workloads_are_bounded_deterministic_and_memory_safe():
    first = generate_realistic_workloads(CONFIG, "bfloat16", GB300_MEMORY_BYTES)
    second = generate_realistic_workloads(CONFIG, "bfloat16", GB300_MEMORY_BYTES)

    assert first == second
    assert Counter(item.phase for item in first.cases) == {
        "discovery": 793,
        "boundary": 300,
        "holdout": 25,
    }
    assert len({(item.phase, item.case.name) for item in first.cases}) == len(
        first.cases
    )
    assert all(
        estimate_case_bytes(item.case, "bfloat16") <= GB300_MEMORY_BYTES * 4 // 5
        for item in first.cases
    )
    assert sum(item.case.has_qv for item in first.cases) == 144
    assert sum(item.case.num_splits < 1 for item in first.cases) == 168
    assert sum(item.case.page_size is not None for item in first.cases) == 74
    assert {item.case.page_size for item in first.cases} == {None, 16, 64, 128}
    assert sum(bool(item.case.score_mod_name) for item in first.cases) == 26
    assert sum(bool(item.case.mask_mod_name) for item in first.cases) == 26
    assert sum(item.case.has_learnable_sink for item in first.cases) == 26
    assert sum(item.case.gather_kv_length is not None for item in first.cases) == 26
    assert (
        sum(item.phase == "holdout" and item.case.has_qv for item in first.cases) == 4
    )
    assert all(reason.startswith("memory:") for reason in first.skipped)


def test_boundary_grid_covers_every_axis_without_full_cartesian_expansion():
    generated = generate_realistic_workloads(CONFIG, "bfloat16", GB300_MEMORY_BYTES)
    boundary = [item.case for item in generated.cases if item.phase == "boundary"]

    assert len({(case.seqlen_q, case.seqlen_k) for case in boundary}) == 120
    assert (
        len({(case.q_heads, case.kv_heads, case.d, case.dv) for case in boundary}) == 10
    )
    assert len({(case.batch, case.causal) for case in boundary}) == 10
    assert {(case.d, case.dv) for case in boundary} == {
        (64, 64),
        (72, 72),
        (80, 80),
        (96, 96),
        (128, 128),
        (192, 128),
        (64, 512),
    }


def test_mla_holdout_uses_packed_qv_shape():
    generated = generate_realistic_workloads(CONFIG, "bfloat16", GB300_MEMORY_BYTES)
    case = next(
        item.case for item in generated.cases if item.case.name == "deepseek_mla_decode"
    )

    assert case.mode == "varlen"
    assert case.has_qv
    assert case.q_heads == 128 and case.kv_heads == 1
    assert case.d == 64 and case.dv == 512
    assert len(case.seqlens_q or []) == 64
    assert set(case.seqlens_q or []) == {1}
    assert set(case.seqlens_k or []) == {32768}


def test_paged_holdouts_cover_standard_and_absorbed_decode():
    generated = generate_realistic_workloads(CONFIG, "bfloat16", GB300_MEMORY_BYTES)
    paged = {
        item.case.name: item.case
        for item in generated.cases
        if item.phase == "holdout" and item.case.page_size is not None
    }

    assert set(paged) == {"qwen2_5_paged_decode", "deepseek_mla_paged_decode"}
    assert paged["qwen2_5_paged_decode"].page_size == 64
    assert paged["deepseek_mla_paged_decode"].page_size == 128
    assert paged["deepseek_mla_paged_decode"].has_qv


def test_modifier_strata_have_independent_model_holdouts():
    generated = generate_realistic_workloads(CONFIG, "bfloat16", GB300_MEMORY_BYTES)
    holdouts = [item.case for item in generated.cases if item.phase == "holdout"]

    assert sum(bool(case.score_mod_name) for case in holdouts) == 2
    assert sum(bool(case.mask_mod_name) for case in holdouts) == 2
    assert sum(case.has_learnable_sink for case in holdouts) == 2
    assert sum(case.gather_kv_length is not None for case in holdouts) == 2


def test_bucket_aggregation_excludes_configs_missing_from_any_case():
    production = {"num_splits": 1}
    common = {"num_splits": 2}
    incomplete = {"num_splits": 16}

    summary = aggregate_bucket_results(
        [
            aggregate_test_row(
                "first", production, common, [("incomplete", incomplete, 1.0)]
            ),
            aggregate_test_row("second", production, common, []),
        ]
    )[0]

    assert summary["cases"] == 2
    assert summary["winner_config"] == common
    assert summary["production_to_winner_speedup"] == 200.0 / 180.0
    assert summary["excluded_incomplete_configs"] == [incomplete]


def test_flex_flash_block_sparse_cases_are_model_held_out():
    cases = generate_flex_block_sparse_cases(CONFIG)
    discovery = [case for case in cases if case.phase == "discovery"]
    holdout = [case for case in cases if case.phase == "holdout"]

    assert len(discovery) == 32
    assert len(holdout) == 6
    assert {case.mask_name for case in cases} == {
        "block_diagonal_128",
        "causal_window_128",
    }
    assert {case.profile for case in holdout}.isdisjoint(
        {case.profile for case in discovery}
    )


def test_model_family_holdouts_are_absent_from_discovery():
    generated = generate_realistic_workloads(CONFIG, "bfloat16", GB300_MEMORY_BYTES)
    discovery_profiles = {
        item.case.profile for item in generated.cases if item.phase == "discovery"
    }
    holdout_profiles = {
        item.case.profile for item in generated.cases if item.phase == "holdout"
    }

    assert {"qwen2_5_7b", "mistral_7b_v03", "falcon_7b", "qwen2_5_vl_vision"} <= (
        holdout_profiles - discovery_profiles
    )
