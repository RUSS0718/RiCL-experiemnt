"""Deterministic offline metrics for MS3 paired repair outputs."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

GROUPS = ("G0", "G1", "G2", "G3")
COMPARISONS = (("G3", "G1"), ("G3", "G2"), ("G1", "G0"))
OUTCOMES = {
    "repair_success": ({"correct": 1, "incorrect": 0}, "benefit"),
    "error_removal": ({"removed": 1, "not_removed": 0}, "benefit"),
    "new_error": ({"present": 1, "absent": 0}, "risk"),
}


def analyze_repairs(
    evaluations: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute paired effects, missingness, clustered intervals, and position layers."""
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative")
    evaluation_by_id = {str(row["evaluation_id"]): dict(row) for row in evaluations}
    assignment_rows = [dict(row) for row in assignments]
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for assignment in assignment_rows:
        evaluation = evaluation_by_id.get(str(assignment["evaluation_id"]))
        if evaluation is None:
            evaluation = {
                "repair_success": "missing",
                "error_removal": "missing",
                "new_error": "missing",
                "missing_reason": "evaluation_record_absent",
            }
        by_sample[str(assignment["sample_id"])][str(assignment["group"])] = {
            "assignment": assignment,
            "evaluation": evaluation,
        }

    repair_comparisons = _comparison_set(by_sample, "repair_success", bootstrap_samples, seed)
    common = {
        sample: values
        for sample, values in by_sample.items()
        if set(values) == set(GROUPS)
        and all(
            _binary(values[group]["evaluation"].get("repair_success")) is not None
            for group in GROUPS
        )
    }
    common_comparisons = _comparison_set(common, "repair_success", bootstrap_samples, seed)
    outcome_reports = {
        outcome: _comparison_set(by_sample, outcome, bootstrap_samples, seed)
        for outcome in ("error_removal", "new_error")
    }
    common_outcome_reports = {
        outcome: _comparison_set(common, outcome, bootstrap_samples, seed)
        for outcome in ("error_removal", "new_error")
    }

    return {
        "comparisons": repair_comparisons,
        "outcomes": outcome_reports,
        "four_group_common": {
            "n_samples": len(common),
            "question_group_count": _cluster_count(common),
            "comparisons": common_comparisons,
            "outcomes": common_outcome_reports,
        },
        "position_offset": _position_report(by_sample, bootstrap_samples, seed),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "unit": "question_group",
            "interval": "percentile 95%",
        },
    }


def _comparison_set(
    by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
    outcome: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        f"{treated}-{control}": _comparison(
            by_sample, treated, control, outcome, bootstrap_samples, seed
        )
        for treated, control in COMPARISONS
    }


def _comparison(
    by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
    treated: str,
    control: str,
    outcome: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    value_map, interpretation = OUTCOMES[outcome]
    pairs = []
    missing: dict[str, int] = defaultdict(int)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id, groups in by_sample.items():
        left = groups.get(treated)
        right = groups.get(control)
        if left is None or right is None:
            missing["missing_group"] += 1
            continue
        left_value = _value(left["evaluation"].get(outcome), value_map)
        right_value = _value(right["evaluation"].get(outcome), value_map)
        stratum = str(left["assignment"].get("stratum") or "unknown")
        if left_value is None or right_value is None:
            if left_value is None:
                missing[f"treated_{_missing_reason(left['evaluation'], outcome, left_value)}"] += 1
            if right_value is None:
                missing[f"control_{_missing_reason(right['evaluation'], outcome, right_value)}"] += 1
            strata[stratum].append({"missing": True})
            continue
        pair = {
            "sample_id": sample_id,
            "question_group_id": left["assignment"].get("question_group_id"),
            "difference": left_value - right_value,
            "treated": left_value,
            "control": right_value,
            "stratum": stratum,
        }
        pairs.append(pair)
        strata[stratum].append(pair)

    if interpretation == "risk":
        changes = {
            "fewer_new_errors": sum(pair["difference"] == -1 for pair in pairs),
            "more_new_errors": sum(pair["difference"] == 1 for pair in pairs),
            "same_with_new_error": sum(
                pair["difference"] == 0 and pair["treated"] == 1 for pair in pairs
            ),
            "same_without_new_error": sum(
                pair["difference"] == 0 and pair["treated"] == 0 for pair in pairs
            ),
        }
    else:
        changes = {
            "improved": sum(pair["difference"] == 1 for pair in pairs),
            "worsened": sum(pair["difference"] == -1 for pair in pairs),
            "same_success_or_removed": sum(
                pair["difference"] == 0 and pair["treated"] == 1 for pair in pairs
            ),
            "same_failure_or_not_removed": sum(
                pair["difference"] == 0 and pair["treated"] == 0 for pair in pairs
            ),
        }
    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "n_pairs": len(pairs),
        "question_group_count": len({str(pair["question_group_id"]) for pair in pairs}),
        "candidate_pairs": len(by_sample),
        "evaluable_rate": len(pairs) / len(by_sample) if by_sample else None,
        "mean_difference": _mean([pair["difference"] for pair in pairs]),
        "counts": changes,
        "missing": dict(missing),
        "by_stratum": {
            name: {
                "candidate_pairs": len(rows),
                "evaluable_pairs": sum("difference" in row for row in rows),
                "missing_pairs": sum("missing" in row for row in rows),
                "mean_difference": _mean(
                    [row["difference"] for row in rows if "difference" in row]
                ),
            }
            for name, rows in sorted(strata.items())
        },
        "cluster_bootstrap_95_ci": _cluster_bootstrap_ci(pairs, bootstrap_samples, seed),
        "pairs": pairs,
    }


def _position_report(
    by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    observations = []
    direction_counts = {"negative": 0, "zero": 0, "positive": 0}
    directions: dict[str, list[str]] = {"earlier": [], "later": [], "same": []}
    offset_sizes: dict[str, list[str]] = defaultdict(list)
    lengths: dict[str, list[str]] = defaultdict(list)
    strata: dict[str, list[str]] = defaultdict(list)
    for sample_id, groups in by_sample.items():
        if "G2" not in groups or "G3" not in groups:
            continue
        g2 = groups["G2"]["assignment"]
        g3 = groups["G3"]["assignment"]
        g2_position = g2.get("g2_position")
        g3_position = g3.get("g3_position", g3.get("first_error_position"))
        if not isinstance(g2_position, int) or not isinstance(g3_position, int):
            continue
        offset = g2_position - g3_position
        direction = "negative" if offset < 0 else "positive" if offset > 0 else "zero"
        layer = "earlier" if offset < 0 else "later" if offset > 0 else "same"
        direction_counts[direction] += 1
        observations.append({
            "sample_id": sample_id,
            "g2_prompt_index": g2_position,
            "g3_prompt_index": g3_position,
            "offset_g2_minus_g3": offset,
            "absolute_offset": abs(offset),
            "trajectory_length": g2.get("trajectory_length"),
            "stratum": g2.get("stratum"),
            "direction": direction,
        })
        directions[layer].append(sample_id)
        magnitude = "1" if abs(offset) == 1 else "2" if abs(offset) == 2 else "3+"
        offset_sizes[magnitude].append(sample_id)
        length = g2.get("trajectory_length")
        lengths[str(length) if isinstance(length, int) else "unknown"].append(sample_id)
        strata[str(g2.get("stratum") or "unknown")].append(sample_id)

    return {
        "direction": direction_counts,
        "position_prompt_pairs": len(observations),
        "observations": observations,
        "layers": {
            layer: _position_layer(by_sample, sample_ids, bootstrap_samples, seed)
            for layer, sample_ids in directions.items()
        },
        "by_absolute_offset": {
            magnitude: _position_layer(by_sample, sample_ids, bootstrap_samples, seed)
            for magnitude, sample_ids in sorted(offset_sizes.items())
        },
        "by_trajectory_length": {
            length: _position_layer(by_sample, sample_ids, bootstrap_samples, seed)
            for length, sample_ids in sorted(lengths.items())
        },
        "by_stratum": {
            name: _position_layer(by_sample, sample_ids, bootstrap_samples, seed)
            for name, sample_ids in sorted(strata.items())
        },
        "empty_layers": [name for name, samples in directions.items() if not samples],
    }


def _position_layer(
    by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sample_ids: list[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = {sample: by_sample[sample] for sample in sample_ids}
    effect = _comparison(selected, "G3", "G2", "repair_success", bootstrap_samples, seed)
    return {
        "position_pairs": len(sample_ids),
        "evaluable_pairs": effect["n_pairs"],
        "mean_difference": effect["mean_difference"],
        "missing": effect["missing"],
        "question_group_count": effect["question_group_count"],
    }


def _cluster_bootstrap_ci(
    pairs: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> dict[str, float] | None:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        clusters[str(pair["question_group_id"])].append(pair)
    names = sorted(clusters)
    if bootstrap_samples == 0 or len(names) < 2:
        return None
    randomizer = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        selected = [randomizer.choice(names) for _ in names]
        values = [row["difference"] for name in selected for row in clusters[name]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return {"lower": _quantile(estimates, 0.025), "upper": _quantile(estimates, 0.975)}


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _missing_reason(evaluation: Mapping[str, Any], outcome: str, value: int | None) -> str:
    if value is not None:
        return "evaluable"
    status = evaluation.get(outcome)
    if status == "unknown":
        return "unknown"
    if status == "missing":
        return str(evaluation.get("missing_reason") or "missing_output")
    return "missing_or_invalid_status"


def _value(status: Any, mapping: Mapping[str, int]) -> int | None:
    return mapping.get(status) if isinstance(status, str) else None


def _binary(value: Any) -> int | None:
    return _value(value, OUTCOMES["repair_success"][0])


def _mean(values: list[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def _cluster_count(by_sample: Mapping[str, Mapping[str, Any]]) -> int:
    return len({
        str(values["G0"]["assignment"].get("question_group_id"))
        for values in by_sample.values()
    })
