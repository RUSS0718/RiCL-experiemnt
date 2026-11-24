"""Offline MS2 route-selection and ranking analysis."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

METHODS = ("rater", "verified_progress", "rater_plus_evidence", "random")
PRIMARY_COMPARISON = "rater_plus_evidence_minus_rater"
VALID_ROLLOUT_STATUSES = {"correct", "incorrect"}


def analyze_ms2_trace(
    trace: Mapping[str, Any] | Path,
    *,
    scoring_dir: Path | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Recompute MS2 main metrics from a persisted trace without model calls."""
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates cannot be negative")
    value = _load_object(trace)
    if scoring_dir is None and isinstance(trace, Path):
        candidate_scoring_dir = trace.parent.parent / "scoring"
        scoring_dir = candidate_scoring_dir if candidate_scoring_dir.is_dir() else None
    scoring_index = _load_scoring_index(scoring_dir)
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("MS2 trace must contain a non-empty samples list")
    prepared: list[dict[str, Any]] = []
    exclusions: dict[str, dict[str, Any]] = {}
    diagnostics = Counter()
    for sample in samples:
        row, reasons, row_diagnostics = _prepare_sample(sample, scoring_index)
        diagnostics.update(row_diagnostics)
        sample_id = _text(sample.get("sample_id"), "sample_id") if isinstance(sample, Mapping) else "unknown"
        if reasons:
            exclusions[sample_id] = {"reasons": reasons}
        else:
            prepared.append(row)
    group_rows = _aggregate_groups(prepared)
    selection_rows = [row for row in group_rows if row["selection_eligible"]]
    ranking_rows = [row for row in group_rows if row["ranking_eligible"]]
    selection = {method: _metric(selection_rows, "selection", method) for method in METHODS}
    ranking = {method: _metric(ranking_rows, "ranking", method) for method in METHODS}
    comparisons = {
        "selection": _comparison_set(selection, selection_rows, "selection", bootstrap_replicates, seed),
        "ranking": _comparison_set(ranking, ranking_rows, "ranking", bootstrap_replicates, seed),
    }
    diagnostics.update({
        "candidate_count": sum(len(row.get("candidates", [])) for row in samples if isinstance(row, Mapping)),
        "analyzable_sample_count": len(prepared),
        "selection_sample_count": len(selection_rows),
        "ranking_sample_count": len(ranking_rows),
    })
    return {
        "schema_version": 1,
        "study": "ms2",
        "run_id": value.get("run_id"),
        "manifest_id": value.get("manifest_id"),
        "manifest_fingerprint": value.get("manifest_fingerprint"),
        "source_count": len({row["source"] for row in prepared}),
        "question_group_count": len({row["question_group_id"] for row in prepared}),
        "summary": {"selection": selection, "ranking": ranking},
        "comparisons": comparisons,
        "coverage": {
            "candidate_samples": len(samples),
            "analyzable_samples": len(prepared),
            "selection_samples": len(selection_rows),
            "ranking_samples": len(ranking_rows),
            "candidate_route_count": sum(len(row["candidates"]) for row in prepared),
            "selection_question_group_count": len({row["question_group_id"] for row in selection_rows}),
            "ranking_question_group_count": len({row["question_group_id"] for row in ranking_rows}),
            "ranking_valid_route_pairs": sum(row["ranking_pair_count"] for row in ranking_rows),
            "denominators": {
                "selection": len(selection_rows),
                "ranking": len(ranking_rows),
                "ranking_pairs": sum(row["ranking_pair_count"] for row in ranking_rows),
            },
        },
        "exclusions": exclusions,
        "diagnostics": dict(diagnostics),
        "secondary": _secondary_diagnostics(prepared),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "unit": "question_group_id",
            "stratification": "source",
            "interval": "percentile 95%",
        },
        "per_group": group_rows,
    }


def write_ms2_analysis(
    trace: Path,
    output: Path,
    *,
    scoring_dir: Path | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 2026,
) -> Path:
    """Write one offline MS2 report and never replace an existing artifact."""
    if output.exists():
        raise ValueError(f"analysis output already exists: {output}")
    report = analyze_ms2_trace(
        trace,
        scoring_dir=scoring_dir,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def _prepare_sample(
    sample: Any, scoring_index: Mapping[tuple[str, str], Mapping[str, Any]]
) -> tuple[dict[str, Any], list[str], Counter[str]]:
    if not isinstance(sample, Mapping):
        raise TypeError("MS2 trace samples must be objects")
    sample_id = _text(sample.get("sample_id"), "sample_id")
    group_id = _text(sample.get("question_group_id") or sample_id, "question_group_id")
    source = _text(sample.get("source") or "unknown", "source")
    candidates = sample.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError(f"{sample_id}: candidates must be a list")
    diagnostics: Counter[str] = Counter()
    parsed = []
    reasons: list[str] = []
    for candidate in candidates:
        route, route_reasons, route_diagnostics = _prepare_candidate(
            candidate,
            scoring_index.get(
                (sample_id, str(candidate.get("route_id") or candidate.get("candidate_id") or "")),
                {},
            ),
        )
        diagnostics.update(route_diagnostics)
        if route_reasons:
            reasons.extend(route_reasons)
        else:
            parsed.append(route)
    if len(parsed) < 2:
        reasons.append("fewer_than_two_eligible_routes")
    row = {
        "sample_id": sample_id,
        "question_group_id": group_id,
        "source": source,
        "candidates": parsed,
    }
    row["selection_eligible"] = not reasons and all(route["scores"] for route in parsed)
    row["ranking_eligible"] = row["selection_eligible"] and any(
        left["y"] != right["y"] for index, left in enumerate(parsed) for right in parsed[index + 1 :]
    )
    row["ranking_pair_count"] = sum(
        left["y"] != right["y"] for index, left in enumerate(parsed) for right in parsed[index + 1 :]
    )
    return row, sorted(set(reasons)), diagnostics


def _prepare_candidate(
    candidate: Any, scoring: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], list[str], Counter[str]]:
    if not isinstance(candidate, Mapping):
        raise TypeError("MS2 candidates must be objects")
    diagnostics: Counter[str] = Counter()
    reasons: list[str] = []
    prefix_status = candidate.get("prefix_status")
    if prefix_status != "available":
        diagnostics[f"prefix_{prefix_status or 'missing'}"] += 1
        reasons.append(f"prefix_{prefix_status or 'missing'}")
    rollouts = candidate.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != 2:
        diagnostics["missing_rollout"] += 1
        reasons.append("expected_two_rollouts")
        rollouts = []
    observations: list[int] = []
    for rollout in rollouts:
        if not isinstance(rollout, Mapping):
            reasons.append("invalid_rollout_record")
            continue
        status = rollout.get("status")
        evaluation = rollout.get("evaluation")
        if status != "completed" or not isinstance(evaluation, Mapping):
            diagnostics["missing_rollout_evaluations"] += 1
            reasons.append("route_missing_independent_evaluation")
            continue
        evaluation_status = evaluation.get("status")
        if evaluation_status not in VALID_ROLLOUT_STATUSES:
            diagnostics["missing_rollout_evaluations"] += 1
            reasons.append("route_missing_independent_evaluation")
            continue
        observations.append(1 if evaluation_status == "correct" else 0)
    if len(observations) != 2:
        reasons.append("route_missing_independent_evaluation")
    scores = _scores(candidate, scoring or {})
    if not scores:
        diagnostics["missing_scores"] += 1
        reasons.append("route_missing_score")
    return {
        "candidate_id": str(candidate.get("candidate_id") or candidate.get("route_id") or "unknown"),
        "y": sum(observations) / 2 if len(observations) == 2 else None,
        "observations": observations,
        "scores": scores,
    }, sorted(set(reasons)), diagnostics


def _scores(
    candidate: Mapping[str, Any], scoring: Mapping[str, Any] | None = None
) -> dict[str, float]:
    raw_candidates = [
        candidate.get("scores"),
        candidate.get("evaluation_scores"),
        candidate.get("score_artifact"),
        candidate.get("scoring_artifact"),
        candidate.get("scoring"),
        scoring,
        candidate,
    ]
    values: dict[str, float] = {}
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            continue
        values.setdefault("rater", _nested_number(raw, ("rater", "llm_rater", "rater_score", "score")))
        values.setdefault(
            "verified_progress",
            _nested_number(raw, ("verified_progress", "progress", "verified_progress_score")),
        )
        values.setdefault("evidence", _nested_number(raw, ("evidence", "evidence_score")))
    values = {name: value for name, value in values.items() if value is not None}
    if "rater" not in values or "verified_progress" not in values:
        return {}
    evidence = values.get("evidence", values["verified_progress"])
    values["rater_plus_evidence"] = (values["rater"] + evidence) / 2
    return values


def _nested_number(raw: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, Mapping):
            value = value.get("score", value.get("value"))
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None



def _aggregate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_group_id"]].append(row)
    result = []
    for group_id, group_rows in sorted(grouped.items()):
        base = group_rows[0]
        if len(group_rows) == 1:
            result.append(base)
            continue
        merged = dict(base)
        merged["replicates"] = group_rows
        merged["selection_eligible"] = all(row["selection_eligible"] for row in group_rows)
        merged["ranking_eligible"] = all(row["ranking_eligible"] for row in group_rows)
        merged["ranking_pair_count"] = sum(row["ranking_pair_count"] for row in group_rows)
        result.append(merged)
    return result



def _metric(rows: list[dict[str, Any]], kind: str, method: str) -> dict[str, Any]:
    values = [_row_metric(row, kind, method) for row in rows]
    values = [value for value in values if value is not None]
    return {
        "value": float(sum(values) / len(values)) if values else None,
        "n": len(values),
        "question_group_count": len(rows),
        "unit": "question_group_id",
    }


def _row_metric(row: Mapping[str, Any], kind: str, method: str) -> float | None:
    replicates = row.get("replicates")
    if isinstance(replicates, list):
        values = [_row_metric(replicate, kind, method) for replicate in replicates]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None
    candidates = row["candidates"]
    if kind == "selection":
        if method == "random":
            return sum(route["y"] for route in candidates) / len(candidates)
        top = _top_routes(candidates, method)
        return sum(route["y"] for route in top) / len(top)
    pair_values = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if left["y"] == right["y"]:
                continue
            if method == "random":
                pair_values.append(0.5)
            else:
                left_score = left["scores"][method]
                right_score = right["scores"][method]
                prediction = 1 if left_score > right_score else 0.5 if left_score == right_score else 0
                actual = 1 if left["y"] > right["y"] else 0
                pair_values.append(prediction if actual else 1 - prediction)
    return sum(pair_values) / len(pair_values) if pair_values else None



def _top_routes(candidates: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    maximum = max(route["scores"][method] for route in candidates)
    return [route for route in candidates if route["scores"][method] == maximum]


def _comparison_set(summary: Mapping[str, Mapping[str, Any]], rows: list[dict[str, Any]], kind: str, replicates: int, seed: int) -> dict[str, Any]:
    result = {}
    for method in ("verified_progress", "rater_plus_evidence", "random"):
        left = summary[method]["value"]
        right = summary["rater"]["value"]
        estimate = None if left is None or right is None else left - right
        ci, reason = _bootstrap_difference(rows, kind, method, replicates, seed)
        result[f"{method}_minus_rater"] = {"estimate": estimate, "ci_95": ci, "ci_reason": reason, "n": len(rows)}
    return result


def _bootstrap_difference(rows: list[dict[str, Any]], kind: str, method: str, replicates: int, seed: int) -> tuple[list[float] | None, str | None]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["question_group_id"])].append(row)
    if replicates == 0:
        return None, "bootstrap_disabled"
    if len(clusters) < 2:
        return None, "fewer_than_two_question_groups"
    by_source: dict[str, list[str]] = defaultdict(list)
    for group_id, group_rows in clusters.items():
        by_source[str(group_rows[0].get("source") or "unknown")].append(group_id)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled_ids = [rng.choice(sorted(ids)) for ids in by_source.values() for _ in ids]
        sampled = [row for group_id in sampled_ids for row in clusters[group_id]]
        left = _metric(sampled, kind, method)["value"]
        right = _metric(sampled, kind, "rater")["value"]
        if left is not None and right is not None:
            draws.append(left - right)
    if not draws:
        return None, "no_valid_bootstrap_replicates"
    values = np.quantile(draws, [0.025, 0.975]).tolist()
    return [float(values[0]), float(values[1])], None


def _secondary_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = []
    for row in rows:
        for candidate in row["candidates"]:
            observations.extend([candidate["y"]] * 2)
    result: dict[str, Any] = {"route_observations": len(observations)}
    for method in ("rater", "verified_progress", "rater_plus_evidence"):
        scores = [candidate["scores"][method] for row in rows for candidate in row["candidates"] for _ in range(2)]
        result[f"{method}_spearman"] = _spearman(scores, observations)
    result["observation_resolution"] = "two independent rollout outcomes per route; y in {0, 0.5, 1}"
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = spearmanr(left, right).statistic
    return None if not np.isfinite(value) else float(value)


def _load_scoring_index(
    scoring_dir: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load persisted #17 route scores by stable sample and route identity."""
    if scoring_dir is None or not scoring_dir.is_dir():
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(scoring_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        sample_id = value.get("sample_id")
        route_id = value.get("route_id")
        if isinstance(sample_id, str) and isinstance(route_id, str):
            index[(sample_id, route_id)] = dict(value)
    return index


def _load_object(value: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    loaded = json.loads(value.read_text(encoding="utf-8")) if isinstance(value, Path) else value
    if not isinstance(loaded, Mapping):
        raise TypeError("MS2 trace must be a JSON object")
    return loaded


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
