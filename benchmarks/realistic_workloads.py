"""Load model-backed FA4 workloads without expanding an unbounded Cartesian grid."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from clc_bench import (
    Case,
    dense_case_name,
    normalize_lengths,
    pattern_weights,
    varlen_case_name,
)

Phase = Literal["discovery", "boundary", "holdout"]


@dataclass(frozen=True)
class GeneratedCase:
    """One benchmark case with its discovery/validation phase."""

    phase: Phase
    case: Case


@dataclass(frozen=True)
class GeneratedWorkloads:
    """Generated cases plus explicit reasons for omitted workload families."""

    cases: tuple[GeneratedCase, ...]
    skipped: tuple[str, ...]


def load_workload_spec(path: Path) -> dict:
    """Load and minimally validate a realistic-workload YAML specification."""
    spec = yaml.safe_load(path.read_text())
    profiles = spec.get("model_profiles", [])
    profile_ids = [profile["id"] for profile in profiles]
    if not profiles or len(profile_ids) != len(set(profile_ids)):
        raise ValueError("Realistic workload profiles must be nonempty and unique")
    holdouts = spec.get("scenario_templates", {}).get("holdout", [])
    holdout_names = [case["name"] for case in holdouts]
    if len(holdout_names) != len(set(holdout_names)):
        raise ValueError("Realistic holdout names must be unique")
    return spec


def estimate_case_bytes(case: Case, dtype_name: str) -> int:
    """Estimate resident Q/K/V/O bytes before allocating a benchmark case."""
    element_size = 1 if dtype_name.startswith("float8") else 2
    if case.mode == "varlen":
        total_q = sum(case.seqlens_q or [])
        lengths_k = case.seqlens_k or case.seqlens_q or []
        total_k = (
            len(lengths_k) * math.ceil(max(lengths_k) / case.page_size) * case.page_size
            if case.page_size is not None
            else sum(lengths_k)
        )
    else:
        total_q = (case.batch or 0) * (case.seqlen_q or 0)
        total_k = (case.batch or 0) * (case.seqlen_k or 0)
    elements = total_q * case.q_heads * (case.d + case.dv) + total_k * case.kv_heads * (
        case.d + case.dv
    )
    if case.has_qv:
        elements += total_q * case.q_heads * case.dv
    return elements * element_size


def profile_matches_dtype(profile: dict, dtype_name: str) -> bool:
    """Keep dtype campaigns independent rather than silently transferring profiles."""
    return profile.get("dtype", "bfloat16") == dtype_name


def within_context(profile: dict, q_length: int, k_length: int) -> bool:
    """Return whether explicit lengths respect a declared model context limit."""
    max_context = profile.get("max_context")
    return max_context is None or max(q_length, k_length) <= max_context


def make_dense_profile_case(
    profile: dict,
    scenario: str,
    batch: int,
    q_length: int,
    k_length: int,
    causal: bool,
    *,
    name: str | None = None,
    num_splits: int = 1,
    score_mod_name: str = "",
    mask_mod_name: str = "",
    has_learnable_sink: bool = False,
    gather_kv_length: int | None = None,
) -> Case:
    """Build one dense case from a public model profile."""
    base_name = dense_case_name(
        profile["q_heads"],
        profile["kv_heads"],
        causal,
        profile["d"],
        profile["dv"],
        batch,
        q_length,
        k_length,
    )
    return Case(
        name=name or f"{profile['id']}__{scenario}__{base_name}",
        mode="dense",
        q_heads=profile["q_heads"],
        kv_heads=profile["kv_heads"],
        d=profile["d"],
        dv=profile["dv"],
        causal=causal,
        batch=batch,
        seqlen_q=q_length,
        seqlen_k=k_length,
        num_splits=num_splits,
        has_qv=profile.get("has_qv", False),
        score_mod_name=score_mod_name,
        mask_mod_name=mask_mod_name,
        has_learnable_sink=has_learnable_sink,
        gather_kv_length=gather_kv_length,
        profile=profile["id"],
        scenario=scenario,
    )


def make_varlen_profile_case(
    profile: dict,
    scenario: str,
    total_q: int,
    total_k: int,
    batch: int,
    pattern: str,
    causal: bool,
    *,
    name: str | None = None,
    page_size: int | None = None,
) -> Case:
    """Build one deterministic packed-varlen case from a model profile."""
    weights = pattern_weights(pattern, batch)
    lengths_q = normalize_lengths(weights, total_q)
    lengths_k = normalize_lengths(weights, total_k)
    return Case(
        name=name
        or (
            f"{profile['id']}__{scenario}__"
            + varlen_case_name(
                pattern,
                profile["q_heads"],
                profile["kv_heads"],
                causal,
                profile["d"],
                profile["dv"],
                batch,
                total_q,
                total_k,
            )
            + (f"__page{page_size}" if page_size is not None else "")
        ),
        mode="varlen",
        q_heads=profile["q_heads"],
        kv_heads=profile["kv_heads"],
        d=profile["d"],
        dv=profile["dv"],
        causal=causal,
        batch=batch,
        seqlens_q=lengths_q,
        seqlens_k=lengths_k,
        pattern=pattern,
        has_qv=profile.get("has_qv", False),
        page_size=page_size,
        profile=profile["id"],
        scenario=scenario,
    )


def generate_realistic_workloads(
    path: Path,
    dtype_name: str,
    total_memory_bytes: int,
) -> GeneratedWorkloads:
    """Expand a bounded realistic corpus and reject unsafe allocations up front."""
    spec = load_workload_spec(path)
    profiles = {profile["id"]: profile for profile in spec["model_profiles"]}
    generated: list[GeneratedCase] = []
    skipped: list[str] = []

    def add(phase: Phase, case: Case) -> None:
        profile = profiles.get(case.profile)
        if profile is not None and not profile_matches_dtype(profile, dtype_name):
            return
        if estimate_case_bytes(case, dtype_name) > total_memory_bytes * 4 // 5:
            skipped.append(f"memory:{case.name}")
            return
        generated.append(GeneratedCase(phase, case))

    discovery = spec["scenario_templates"]["discovery"]
    full = discovery["full_prefill_training"]
    for profile_id in full["profiles"]:
        profile = profiles[profile_id]
        for context in full["contexts"]:
            if not within_context(profile, context, context):
                continue
            for batch in full["batches_by_context"][context]:
                add(
                    "discovery",
                    make_dense_profile_case(
                        profile, "full_prefill_training", batch, context, context, True
                    ),
                )

    chunked = discovery["chunked_prefill"]
    for profile_id in chunked["profiles"]:
        profile = profiles[profile_id]
        for q_length in chunked["q_lengths"]:
            for k_length in chunked["k_lengths"]:
                if q_length > k_length or not within_context(
                    profile, q_length, k_length
                ):
                    continue
                for batch in chunked["batches_by_k"][k_length]:
                    add(
                        "discovery",
                        make_dense_profile_case(
                            profile,
                            "chunked_prefill",
                            batch,
                            q_length,
                            k_length,
                            True,
                        ),
                    )

    decode = discovery["decode"]
    for profile_id in decode["profiles"]:
        profile = profiles[profile_id]
        for q_length in decode["q_lengths"]:
            for k_length in decode["k_lengths"]:
                if not within_context(profile, q_length, k_length):
                    continue
                for batch in decode["batches_by_k"][k_length]:
                    add(
                        "discovery",
                        make_dense_profile_case(
                            profile,
                            "decode",
                            batch,
                            q_length,
                            k_length,
                            True,
                            num_splits=0,
                        ),
                    )

    paged = discovery["paged_decode"]
    for profile_id in paged["profiles"]:
        profile = profiles[profile_id]
        for q_length in paged["q_lengths"]:
            for k_length in paged["k_lengths"]:
                if not within_context(profile, q_length, k_length):
                    continue
                for batch in paged["batches_by_k"][k_length]:
                    for page_size in paged["page_sizes"]:
                        add(
                            "discovery",
                            make_varlen_profile_case(
                                profile,
                                "paged_decode",
                                batch * q_length,
                                batch * k_length,
                                batch,
                                "uniform",
                                True,
                                page_size=page_size,
                            ),
                        )

    noncausal = discovery["noncausal_and_cross"]
    for raw_case in noncausal["cases"]:
        profile = profiles[raw_case["profile"]]
        for batch in raw_case["batches"]:
            add(
                "discovery",
                make_dense_profile_case(
                    profile,
                    "noncausal_and_cross",
                    batch,
                    raw_case["q"],
                    raw_case["k"],
                    False,
                ),
            )

    features = discovery["feature_strata"]
    for feature in features:
        for profile_id in feature["profiles"]:
            profile = profiles[profile_id]
            for q_length, k_length in feature["qk"]:
                if not within_context(profile, q_length, k_length):
                    continue
                for batch in feature["batches"]:
                    add(
                        "discovery",
                        make_dense_profile_case(
                            profile,
                            f"feature_{feature['name']}",
                            batch,
                            q_length,
                            k_length,
                            feature.get("causal", False),
                            score_mod_name=feature.get("score_mod_name", ""),
                            mask_mod_name=feature.get("mask_mod_name", ""),
                            has_learnable_sink=feature.get("has_learnable_sink", False),
                        ),
                    )

    gather = discovery["gather_kv"]
    for profile_id in gather["profiles"]:
        profile = profiles[profile_id]
        for q_length in gather["q_lengths"]:
            for k_length in gather["k_lengths"]:
                for batch in gather["batches"]:
                    for gather_kv_length in gather["gather_lengths"]:
                        add(
                            "discovery",
                            make_dense_profile_case(
                                profile,
                                "feature_gather_kv",
                                batch,
                                q_length,
                                k_length,
                                True,
                                name=(
                                    f"{profile['id']}__feature_gather_kv__"
                                    + dense_case_name(
                                        profile["q_heads"],
                                        profile["kv_heads"],
                                        True,
                                        profile["d"],
                                        profile["dv"],
                                        batch,
                                        q_length,
                                        k_length,
                                    )
                                    + f"__gather{gather_kv_length}"
                                ),
                                gather_kv_length=gather_kv_length,
                            ),
                        )

    varlen = discovery["packed_varlen"]
    for profile_id in varlen["profiles"]:
        profile = profiles[profile_id]
        for total_q, total_k in varlen["total_qk"]:
            for batch in varlen["batches"]:
                for pattern in varlen["patterns"]:
                    for causal in varlen["causal_values"]:
                        case = make_varlen_profile_case(
                            profile,
                            "packed_varlen",
                            total_q,
                            total_k,
                            batch,
                            pattern,
                            causal,
                        )
                        if within_context(
                            profile,
                            max(case.seqlens_q or [0]),
                            max(case.seqlens_k or [0]),
                        ):
                            add("discovery", case)

    mla = discovery["mla_and_diff_head"]
    for profile_id in mla["profiles"]:
        profile = profiles[profile_id]
        for q_length, k_length in mla["qk"]:
            if not within_context(profile, q_length, k_length):
                continue
            for batch in mla["batches"]:
                add(
                    "discovery",
                    make_dense_profile_case(
                        profile,
                        "mla_and_diff_head",
                        batch,
                        q_length,
                        k_length,
                        True,
                    ),
                )

    boundary = spec["scenario_templates"]["boundary"]
    boundary_budget = spec["case_budgets"]["boundary"]
    axes = (
        boundary["q_lengths"],
        boundary["k_lengths"],
        boundary["batches"],
        boundary["head_dim_regimes"],
    )
    # Generate a small deterministic reserve so memory-gated cases do not shrink the budget.
    for index in range(boundary_budget + len(axes[3])):
        num_qk_pairs = len(axes[0]) * len(axes[1])
        qk_index = index * 37 % num_qk_pairs
        cycle = index // num_qk_pairs
        q_length = axes[0][qk_index // len(axes[1])]
        k_length = axes[1][qk_index % len(axes[1])]
        batch = axes[2][(index * 3 + cycle) % len(axes[2])]
        q_heads, kv_heads, d, dv, has_qv = axes[3][
            (index * 7 + cycle * 3) % len(axes[3])
        ]
        causal = bool((index + cycle) % 2)
        profile = {
            "id": "synthetic_boundary",
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "d": d,
            "dv": dv,
            "has_qv": has_qv,
        }
        add(
            "boundary",
            make_dense_profile_case(
                profile,
                "pairwise_boundary",
                batch,
                q_length,
                k_length,
                causal,
            ),
        )

    for raw_case in spec["scenario_templates"]["holdout"]:
        profile = profiles[raw_case["profile"]]
        if raw_case["mode"] == "varlen":
            case = make_varlen_profile_case(
                profile,
                "model_holdout",
                raw_case["batch"] * raw_case["q"],
                raw_case["batch"] * raw_case["k"],
                raw_case["batch"],
                "uniform",
                raw_case["causal"],
                name=raw_case["name"],
                page_size=raw_case.get("page_size"),
            )
        else:
            case = make_dense_profile_case(
                profile,
                "model_holdout",
                raw_case["batch"],
                raw_case["q"],
                raw_case["k"],
                raw_case["causal"],
                name=raw_case["name"],
                score_mod_name=raw_case.get("score_mod_name", ""),
                mask_mod_name=raw_case.get("mask_mod_name", ""),
                has_learnable_sink=raw_case.get("has_learnable_sink", False),
                gather_kv_length=raw_case.get("gather_kv_length"),
            )
        add("holdout", case)

    unique = {}
    for item in generated:
        key = (
            item.phase,
            item.case.mode,
            item.case.q_heads,
            item.case.kv_heads,
            item.case.d,
            item.case.dv,
            item.case.causal,
            item.case.has_qv,
            item.case.page_size,
            item.case.score_mod_name,
            item.case.mask_mod_name,
            item.case.has_learnable_sink,
            item.case.gather_kv_length,
            item.case.num_splits,
            item.case.batch,
            item.case.seqlen_q,
            item.case.seqlen_k,
            tuple(item.case.seqlens_q or []),
            tuple(item.case.seqlens_k or []),
        )
        unique.setdefault(key, item)

    phase_budgets = spec["case_budgets"]
    selected = []
    for phase in ("discovery", "boundary", "holdout"):
        phase_cases = [item for item in unique.values() if item.phase == phase]
        selected.extend(phase_cases[: phase_budgets[phase]])
    return GeneratedWorkloads(tuple(selected), tuple(sorted(set(skipped))))
