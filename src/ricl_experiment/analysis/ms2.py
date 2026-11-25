"""Offline MS2 route-selection and ranking analysis."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
from scipy.stats import spearmanr

METHODS = ("rater", "verified_progress", "rater_plus_evidence", "random")
PRIMARY_COMPARISON = "rater_plus_evidence_minus_rater"
VALID_ROLLOUT_STATUSES = {"correct", "incorrect"}


def route_flow_status(candidate: Mapping[str, Any]) -> str:
    """Identify which Checkpoints can enter the shared MS2 route comparison."""
    prefix_status = candidate.get("prefix_status")
    if prefix_status != "available":
        return f"prefix_{prefix_status if isinstance(prefix_status, str) else 'missing'}"
    if candidate.get("checkpoint_review_required"):
        verdict = (candidate.get("checkpoint_review") or {}).get("verdict")
        return {
            "correct": "early_correct",
            "incorrect": "eligible",
            "unknown": "early_unknown",
            "no_final_answer": "eligible",
        }.get(verdict, "checkpoint_review_pending")
    evaluation = candidate.get("initial_evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    category = evaluation.get("analysis_category")
    if category == "early_correct" or evaluation.get("status") == "correct":
        return "early_correct"
    if (category == "early_unknown" or category is None) and evaluation.get("extracted_answer") is not None:
        return "early_unknown"
    return "eligible"


def analyze_ms2_trace(
    trace: Mapping[str, Any] | Path,
    *,
    scoring_dir: Path | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Recompute MS2 main metrics from persisted traces without model calls."""
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates cannot be negative")
    value = _load_object(trace)
    if scoring_dir is None and isinstance(trace, Path):
        candidate_scoring_dir = trace.parent.parent / "scoring"
        scoring_dir = candidate_scoring_dir if candidate_scoring_dir.is_dir() else None
    scoring_index = _load_scoring_index(scoring_dir)
    raw_samples = value.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("MS2 trace must contain a non-empty samples list")
    prepared = []
    diagnostics = Counter()
    expected_scores = valid_scores = expected_rollouts = valid_rollouts = 0
    valid_rater_scores = valid_progress_scores = 0
    evidence_scores: list[int] = []
    sample_ids = []
    for sample in raw_samples:
        row, _reasons, row_diagnostics = _prepare_sample(sample, scoring_index)
        prepared.append(row)
        sample_ids.append(row["sample_id"])
        diagnostics.update(row_diagnostics)
        expected_scores += row["expected_score_slots"]
        valid_scores += row["valid_score_slots"]
        valid_rater_scores += row["valid_rater_score_slots"]
        valid_progress_scores += row["valid_progress_score_slots"]
        expected_rollouts += row["expected_rollout_slots"]
        valid_rollouts += row["valid_rollout_slots"]
        evidence_scores.extend(row["evidence_scores"])
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("MS2 trace sample IDs must be unique")
    group_sources: dict[str, set[str]] = defaultdict(set)
    for row in prepared:
        group_sources[row["question_group_id"]].add(row["source"])
    if any(len(sources) != 1 for sources in group_sources.values()):
        raise ValueError("each question group must belong to exactly one source")

    analyzed_samples = [row for row in prepared if row["selection_eligible"]]
    group_rows = _aggregate_groups(analyzed_samples)
    selection_rows = [row for row in group_rows if row["selection_eligible"]]
    ranking_rows = [row for row in group_rows if row["ranking_eligible"]]
    selection = {method: _metric(selection_rows, "selection", method) for method in METHODS}
    ranking = {method: _metric(ranking_rows, "ranking", method) for method in METHODS}
    comparisons = {
        "selection": _comparison_set(
            selection, selection_rows, "selection", bootstrap_replicates, seed
        ),
        "ranking": _comparison_set(ranking, ranking_rows, "ranking", bootstrap_replicates, seed),
    }
    diagnostics.update(
        {
            "candidate_count": sum(len(row["route_records"]) for row in prepared),
            "analyzable_sample_count": len(analyzed_samples),
            "selection_sample_count": len(selection_rows),
            "ranking_sample_count": len(ranking_rows),
        }
    )
    coverage = {
        "candidate_samples": len(prepared),
        "analyzable_samples": len(analyzed_samples),
        "selection_samples": len(selection_rows),
        "ranking_samples": len(ranking_rows),
        "candidate_route_count": sum(len(row["route_records"]) for row in prepared),
        "analyzable_candidate_route_count": sum(
            len(row["candidates"]) for row in analyzed_samples
        ),
        "selection_question_group_count": len(selection_rows),
        "ranking_question_group_count": len(ranking_rows),
        "ranking_valid_route_pairs": sum(row["ranking_pair_count"] for row in ranking_rows),
        "expected_score_slots": expected_scores,
        "valid_score_slots": valid_scores,
        "missing_score_slots": expected_scores - valid_scores,
        "score_missing_rate": _rate(expected_scores - valid_scores, expected_scores),
        "score_slots_by_method": _score_slot_summary(
            expected_scores // 2, valid_rater_scores, valid_progress_scores
        ),
        "expected_rollout_evaluations": expected_rollouts,
        "valid_rollout_evaluations": valid_rollouts,
        "missing_rollout_evaluations": expected_rollouts - valid_rollouts,
        "rollout_evaluation_missing_rate": _rate(
            expected_rollouts - valid_rollouts, expected_rollouts
        ),
        "denominators": {
            "selection": len(selection_rows),
            "ranking": len(ranking_rows),
            "ranking_pairs": sum(row["ranking_pair_count"] for row in ranking_rows),
        },
    }
    exclusions = {
        row["sample_id"]: {"reasons": row["reasons"]}
        for row in prepared
        if not row["selection_eligible"]
    }
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
        "coverage": coverage,
        "exclusions": exclusions,
        "diagnostics": dict(diagnostics),
        "secondary": _secondary_diagnostics(analyzed_samples),
        "evidence_coverage": _evidence_coverage(prepared, evidence_scores),
        "by_source": _source_summary(prepared, group_rows),
        "costs": _cost_summary(prepared),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "unit": "question_group_id",
            "stratification": "source",
            "interval": "percentile 95%",
        },
        "per_sample": [_sample_result(row) for row in prepared],
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
    """Write an offline JSON analysis and source/evidence coverage chart."""
    figure = output.with_name(f"{output.stem}-coverage.png")
    if output.exists() or figure.exists():
        raise ValueError(f"analysis output already exists: {output} or {figure}")
    report = analyze_ms2_trace(
        trace,
        scoring_dir=scoring_dir,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    figure_temporary = figure.with_name(f"{figure.stem}.tmp.png")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_coverage_figure(report, figure_temporary)
    temporary.replace(output)
    figure_temporary.replace(figure)
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
    route_records = []
    qualified = []
    evidence_scores = []
    expected_score_slots = valid_score_slots = 0
    valid_rater_score_slots = valid_progress_score_slots = 0
    expected_rollout_slots = valid_rollout_slots = 0
    route_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("MS2 candidates must be objects")
        route_id = str(candidate.get("route_id") or candidate.get("candidate_id") or "")
        if not route_id or route_id in route_ids:
            raise ValueError(f"{sample_id}: route IDs must be unique within a sample")
        route_ids.add(route_id)
        scoring = scoring_index.get((sample_id, route_id), {})
        route, route_reasons, route_diagnostics = _prepare_candidate(candidate, scoring)
        diagnostics.update(route_diagnostics)
        if route["prefix_status"] == "available":
            expected_score_slots += 2
            valid_rater_score_slots += int("rater" in route["scores"])
            valid_progress_score_slots += int("verified_progress" in route["scores"])
            valid_score_slots += int("rater" in route["scores"]) + int(
                "verified_progress" in route["scores"]
            )
            expected_rollout_slots += 2
            valid_rollout_slots += min(len(route["observations"]), 2)
            if "verified_progress" in route["scores"]:
                evidence_scores.append(int(route["scores"]["verified_progress"]))
        if route["flow_status"] == "eligible":
            route["reasons"] = route_reasons
            qualified.append(route)
            if route_reasons:
                diagnostics["incomplete_eligible_routes"] += 1
        route_records.append(route)
    reasons = sorted({reason for route in route_records for reason in route.get("reasons", [])})
    if any(route["flow_status"] == "checkpoint_review_pending" for route in route_records):
        reasons.append("checkpoint_review_pending")
    if len(qualified) < 2:
        reasons.append("fewer_than_two_eligible_routes")
    parsed = [route for route in qualified if not route.get("reasons")]
    if not reasons and len(parsed) >= 2:
        _add_composite_ranks(parsed)
    selection_eligible = not reasons and len(parsed) >= 2
    ranking_pair_count = sum(
        left["y"] != right["y"]
        for index, left in enumerate(parsed)
        for right in parsed[index + 1 :]
    )
    row = {
        "sample_id": sample_id,
        "question_group_id": group_id,
        "source": source,
        "candidates": parsed if selection_eligible else [],
        "route_records": route_records,
        "eligible_route_count": len(qualified),
        "selection_eligible": selection_eligible,
        "ranking_eligible": selection_eligible and ranking_pair_count > 0,
        "ranking_pair_count": ranking_pair_count if selection_eligible else 0,
        "expected_score_slots": expected_score_slots,
        "valid_score_slots": valid_score_slots,
        "valid_rater_score_slots": valid_rater_score_slots,
        "valid_progress_score_slots": valid_progress_score_slots,
        "expected_rollout_slots": expected_rollout_slots,
        "valid_rollout_slots": valid_rollout_slots,
        "evidence_scores": evidence_scores,
        "reasons": sorted(set(reasons)),
    }
    diagnostics["included_samples" if selection_eligible else "excluded_samples"] += 1
    return row, row["reasons"], diagnostics


def _prepare_candidate(
    candidate: Any, scoring: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], list[str], Counter[str]]:
    if not isinstance(candidate, Mapping):
        raise TypeError("MS2 candidates must be objects")
    diagnostics: Counter[str] = Counter()
    reasons: list[str] = []
    prefix_status = candidate.get("prefix_status")
    prefix_status = prefix_status if isinstance(prefix_status, str) else "missing"
    evaluation = candidate.get("initial_evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    initial_status = evaluation.get("status") or "unknown"
    flow_status = route_flow_status(candidate)
    diagnostics[flow_status if flow_status.startswith("prefix_") else f"{flow_status}_routes"] += 1

    rollouts = candidate.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != 2:
        diagnostics["missing_rollouts"] += 1
        reasons.append("expected_two_rollouts")
        rollouts = []
    observations: list[int] = []
    rollout_statuses = []
    rollout_usages = []
    for rollout in rollouts:
        if not isinstance(rollout, Mapping):
            diagnostics["invalid_rollout_records"] += 1
            reasons.append("invalid_rollout_record")
            rollout_statuses.append("invalid")
            continue
        rollout_statuses.append(str(rollout.get("status") or "missing"))
        usage = rollout.get("usage")
        rollout_usages.append(dict(usage) if isinstance(usage, Mapping) else None)
        evaluation = rollout.get("evaluation")
        if rollout.get("status") != "completed" or not isinstance(evaluation, Mapping):
            diagnostics["missing_rollout_evaluations"] += 1
            reasons.append("route_missing_independent_evaluation")
            continue
        evaluation_status = evaluation.get("status")
        if evaluation_status not in VALID_ROLLOUT_STATUSES:
            diagnostics["missing_rollout_evaluations"] += 1
            diagnostics[f"rollout_evaluation_{evaluation_status or 'missing'}"] += 1
            reasons.append("route_missing_independent_evaluation")
            continue
        observations.append(1 if evaluation_status == "correct" else 0)
    if len(observations) != 2:
        reasons.append("route_missing_independent_evaluation")

    scores, invalid_scores = _scores(candidate, scoring or {})
    for name in ("rater", "verified_progress"):
        if name not in scores and prefix_status == "available":
            diagnostics[f"missing_{name}_scores"] += 1
            if flow_status == "eligible":
                reasons.append(f"route_missing_{name}_score")
            if name in invalid_scores:
                diagnostics[f"invalid_{name}_scores"] += 1
    if prefix_status == "available" and (
        "rater" not in scores or "verified_progress" not in scores
    ):
        diagnostics["missing_scores"] += 1
    if flow_status != "eligible":
        reasons = []
    return (
        {
            "candidate_id": str(
                candidate.get("candidate_id") or candidate.get("route_id") or "unknown"
            ),
            "route_id": str(
                candidate.get("route_id") or candidate.get("candidate_id") or "unknown"
            ),
            "prefix_status": prefix_status,
            "initial_status": initial_status,
            "flow_status": flow_status,
            "rollout_statuses": rollout_statuses,
            "rollout_usages": rollout_usages,
            "score_cost": _route_score_cost(candidate, scoring),
            "evidence": _route_evidence(candidate, scoring),
            "y": sum(observations) / 2 if len(observations) == 2 else None,
            "observations": observations,
            "scores": scores,
            "invalid_scores": sorted(invalid_scores),
            "reasons": sorted(set(reasons)),
        },
        sorted(set(reasons)),
        diagnostics,
    )


def _scores(
    candidate: Mapping[str, Any], scoring: Mapping[str, Any] | None = None
) -> tuple[dict[str, float], set[str]]:
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
    invalid: set[str] = set()
    fields = {
        "rater": ("rater", "llm_rater", "rater_score", "score"),
        "verified_progress": ("verified_progress", "progress", "verified_progress_score"),
    }
    for name, keys in fields.items():
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                continue
            found, value = _nested_value(raw, keys)
            if not found or value is None:
                continue
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3:
                values[name] = float(value)
                break
            invalid.add(name)
    return values, invalid


def _nested_value(raw: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[bool, Any]:
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, Mapping):
            if "score" in value:
                value = value["score"]
            elif "value" in value:
                value = value["value"]
        return True, value
    return False, None


def _route_score_cost(
    candidate: Mapping[str, Any], scoring: Mapping[str, Any]
) -> dict[str, Any] | None:
    for raw in (scoring, candidate.get("scoring_artifact"), candidate.get("score_artifact")):
        if isinstance(raw, Mapping) and isinstance(raw.get("cost"), Mapping):
            return dict(raw["cost"])
    return None


def _route_evidence(
    candidate: Mapping[str, Any], scoring: Mapping[str, Any]
) -> dict[str, Any] | None:
    for raw in (scoring, candidate.get("scoring_artifact"), candidate.get("score_artifact")):
        if not isinstance(raw, Mapping):
            continue
        progress = raw.get("verified_progress")
        if isinstance(progress, Mapping):
            return {
                key: progress.get(key)
                for key in (
                    "coverage",
                    "coverage_status",
                    "verified",
                    "falsified",
                    "unknown",
                    "evidence_by_source",
                )
            }
    return None


def _add_composite_ranks(candidates: list[dict[str, Any]]) -> None:
    for field in ("rater", "verified_progress"):
        ordered = sorted(
            ((route["scores"][field], index) for index, route in enumerate(candidates)),
            reverse=True,
        )
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
                end += 1
            average_rank = ((cursor + 1) + end) / 2
            for _, index in ordered[cursor:end]:
                candidates[index].setdefault("ranks", {})[field] = average_rank
            cursor = end
    for route in candidates:
        ranks = route["ranks"]
        route["ranks"]["rater_plus_evidence"] = (ranks["rater"] + ranks["verified_progress"]) / 2


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
        if {row["source"] for row in group_rows} != {base["source"]}:
            raise ValueError(f"question group {group_id} has multiple data sources")
        merged = dict(base)
        merged["replicates"] = group_rows
        merged["selection_eligible"] = any(row["selection_eligible"] for row in group_rows)
        merged["ranking_eligible"] = any(row["ranking_eligible"] for row in group_rows)
        merged["ranking_pair_count"] = sum(
            row["ranking_pair_count"] for row in group_rows if row["ranking_eligible"]
        )
        result.append(merged)
    return result


def _metric(rows: list[dict[str, Any]], kind: str, method: str) -> dict[str, Any]:
    values = [_row_metric(row, kind, method) for row in rows]
    values = [value for value in values if value is not None]
    return {
        "value": float(sum(values) / len(values)) if values else None,
        "n": len(values),
        "question_group_count": len(rows),
        "denominator": len(values),
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
                continue
            left_score = (
                left["ranks"][method] if method == "rater_plus_evidence" else left["scores"][method]
            )
            right_score = (
                right["ranks"][method]
                if method == "rater_plus_evidence"
                else right["scores"][method]
            )
            if method == "rater_plus_evidence":
                prediction = (
                    1 if left_score < right_score else 0.5 if left_score == right_score else 0
                )
            else:
                prediction = (
                    1 if left_score > right_score else 0.5 if left_score == right_score else 0
                )
            actual = 1 if left["y"] > right["y"] else 0
            pair_values.append(prediction if actual else 1 - prediction)
    return sum(pair_values) / len(pair_values) if pair_values else None


def _top_routes(candidates: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method == "rater_plus_evidence":
        best = min(route["ranks"][method] for route in candidates)
        return [route for route in candidates if route["ranks"][method] == best]
    maximum = max(route["scores"][method] for route in candidates)
    return [route for route in candidates if route["scores"][method] == maximum]


def _comparison_set(
    summary: Mapping[str, Mapping[str, Any]],
    rows: list[dict[str, Any]],
    kind: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for method in ("verified_progress", "rater_plus_evidence", "random"):
        left = summary[method]["value"]
        right = summary["rater"]["value"]
        estimate = None if left is None or right is None else left - right
        ci, reason = _bootstrap_difference(rows, kind, method, replicates, seed)
        groups_by_source: dict[str, int] = Counter(
            str(row.get("source") or "unknown") for row in rows
        )
        result[f"{method}_minus_rater"] = {
            "estimate": estimate,
            "ci_95": ci,
            "ci_reason": reason,
            "ci_applicable": ci is not None,
            "denominator": len(rows),
            "unit": "question_group_id",
            "scale": "proportion",
            "groups_by_source": groups_by_source,
        }
    return result


def _bootstrap_difference(
    rows: list[dict[str, Any]], kind: str, method: str, replicates: int, seed: int
) -> tuple[list[float] | None, str | None]:
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
        sampled_ids = [
            rng.choice(sorted(by_source[source]))
            for source in sorted(by_source)
            for _ in by_source[source]
        ]
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
    route_labels = [candidate["y"] for row in rows for candidate in row["candidates"]]
    route_scores: dict[str, list[float]] = {}
    rollout_scores: dict[str, list[float]] = {}
    rollout_labels = [
        outcome
        for row in rows
        for candidate in row["candidates"]
        for outcome in candidate["observations"]
    ]
    for method in ("rater", "verified_progress", "rater_plus_evidence"):
        scores = [
            (
                -candidate["ranks"]["rater_plus_evidence"]
                if method == "rater_plus_evidence"
                else candidate["scores"][method]
            )
            for row in rows
            for candidate in row["candidates"]
        ]
        route_scores[method] = scores
        rollout_scores[method] = [
            score
            for row in rows
            for candidate in row["candidates"]
            for score in [
                (
                    -candidate["ranks"]["rater_plus_evidence"]
                    if method == "rater_plus_evidence"
                    else candidate["scores"][method]
                )
            ]
            for _ in candidate["observations"]
        ]
    result: dict[str, Any] = {
        "route_observations": len(rollout_labels),
        "route_count": len(route_labels),
    }
    for method, scores in route_scores.items():
        result[f"{method}_spearman"] = _spearman(scores, route_labels)
        result[f"{method}_auroc"] = _auroc(rollout_scores[method], rollout_labels)
    result["ranking_disagreement"] = _ranking_disagreement(rows)
    result["top_ranked_failure_rate"] = {
        method: _top_ranked_failure(rows, method) for method in METHODS if method != "random"
    }
    result["observation_resolution"] = (
        "two independent rollout outcomes per route; y in {0, 0.5, 1}"
    )
    return result


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1 if positive > negative else 0.5 if positive == negative else 0
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def _ranking_disagreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    disagreements = 0
    tie_disagreements = 0
    pairs = 0
    for row in rows:
        candidates = row["candidates"]
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                pairs += 1
                rater = _order(left["scores"]["rater"], right["scores"]["rater"])
                evidence = _order(
                    left["scores"]["verified_progress"],
                    right["scores"]["verified_progress"],
                )
                disagreements += rater != evidence
                tie_disagreements += (rater == 0) != (evidence == 0)
    return {
        "disagreements": disagreements,
        "tie_disagreements": tie_disagreements,
        "denominator": pairs,
        "rate": _rate(disagreements, pairs),
        "tie_rate": _rate(tie_disagreements, pairs),
    }


def _order(left: float, right: float) -> int:
    return 1 if left > right else -1 if left < right else 0


def _top_ranked_failure(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = []
    for row in rows:
        top = _top_routes(row["candidates"], method)
        values.append(sum(route["y"] == 0 for route in top) / len(top))
    return {
        "value": float(sum(values) / len(values)) if values else None,
        "denominator": len(values),
        "unit": "question_sample",
    }


def _score_slot_summary(
    expected_per_method: int, valid_rater: int, valid_progress: int
) -> dict[str, dict[str, int | float | None]]:
    return {
        method: {
            "expected": expected_per_method,
            "valid": valid,
            "missing": expected_per_method - valid,
            "missing_rate": _rate(expected_per_method - valid, expected_per_method),
        }
        for method, valid in (("rater", valid_rater), ("verified_progress", valid_progress))
    }

def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sample_result(row: Mapping[str, Any]) -> dict[str, Any]:
    route_records = row["route_records"]
    return {
        "sample_id": row["sample_id"],
        "question_group_id": row["question_group_id"],
        "source": row["source"],
        "status": "included" if row["selection_eligible"] else "excluded",
        "reasons": row["reasons"],
        "candidate_count": len(route_records),
        "eligible_route_count": row["eligible_route_count"],
        "early_correct_route_count": sum(
            r["flow_status"] == "early_correct" for r in route_records
        ),
        "early_unknown_route_count": sum(
            r["flow_status"] == "early_unknown" for r in route_records
        ),
        "selection_eligible": row["selection_eligible"],
        "ranking_eligible": row["ranking_eligible"],
        "ranking_pair_count": row["ranking_pair_count"],
        "selection": {
            method: _row_metric(row, "selection", method) if row["selection_eligible"] else None
            for method in METHODS
        },
        "ranking": {
            method: _row_metric(row, "ranking", method) if row["ranking_eligible"] else None
            for method in METHODS
        },
        "paired_difference": {
            kind: (
                _row_metric(row, kind, "rater_plus_evidence") - _row_metric(row, kind, "rater")
                if row["selection_eligible"] and (kind == "selection" or row["ranking_eligible"])
                else None
            )
            for kind in ("selection", "ranking")
        },
        "routes": route_records,
    }


def _evidence_coverage(samples: list[dict[str, Any]], all_scores: list[int]) -> dict[str, Any]:
    distribution = Counter(str(score) for score in all_scores)
    tied_pairs = eligible_pairs = all_tied_samples = 0
    status_counts: Counter[str] = Counter()
    claim_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for row in samples:
        for route in row["route_records"]:
            evidence = route.get("evidence")
            if route["prefix_status"] != "available":
                continue
            if not isinstance(evidence, Mapping):
                status_counts["missing"] += 1
                continue
            status_counts[str(evidence.get("coverage_status") or "missing")] += 1
            by_source = evidence.get("evidence_by_source")
            if isinstance(by_source, Mapping):
                for source, counts in by_source.items():
                    if isinstance(counts, Mapping):
                        for verdict in ("verified", "falsified", "unknown"):
                            claim_sources[str(source)][verdict] += int(counts.get(verdict, 0) or 0)
        scores = [
            route["scores"]["verified_progress"]
            for route in row["route_records"]
            if route["flow_status"] == "eligible" and "verified_progress" in route["scores"]
        ]
        pairs = [
            (left, right) for index, left in enumerate(scores) for right in scores[index + 1 :]
        ]
        equal = sum(left == right for left, right in pairs)
        eligible_pairs += len(pairs)
        tied_pairs += equal
        if row["selection_eligible"] and len(scores) >= 2 and equal == len(pairs):
            all_tied_samples += 1
    return {
        "score_distribution": dict(sorted(distribution.items(), key=lambda item: int(item[0]))),
        "tied_pairs": tied_pairs,
        "eligible_pairs": eligible_pairs,
        "tie_rate": _rate(tied_pairs, eligible_pairs),
        "all_tied_sample_count": all_tied_samples,
        "coverage_status_counts": dict(sorted(status_counts.items())),
        "claims_by_source": {
            source: dict(counts) for source, counts in sorted(claim_sources.items())
        },
    }


def _source_summary(samples: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in sorted({row["source"] for row in samples}):
        source_samples = [row for row in samples if row["source"] == source]
        source_groups = [row for row in groups if row["source"] == source]
        selection = [row for row in source_groups if row["selection_eligible"]]
        ranking = [row for row in source_groups if row["ranking_eligible"]]
        result[source] = {
            "candidate_samples": len(source_samples),
            "expected_score_slots": sum(row["expected_score_slots"] for row in source_samples),
            "valid_score_slots": sum(row["valid_score_slots"] for row in source_samples),
            "score_slots_by_method": _score_slot_summary(
                sum(row["expected_score_slots"] for row in source_samples) // 2,
                sum(row["valid_rater_score_slots"] for row in source_samples),
                sum(row["valid_progress_score_slots"] for row in source_samples),
            ),
            "expected_rollout_evaluations": sum(
                row["expected_rollout_slots"] for row in source_samples
            ),
            "valid_rollout_evaluations": sum(row["valid_rollout_slots"] for row in source_samples),
            "included_samples": sum(row["selection_eligible"] for row in source_samples),
            "excluded_samples": sum(not row["selection_eligible"] for row in source_samples),
            "selection_question_groups": len(selection),
            "ranking_question_groups": len(ranking),
            "evidence_coverage": _evidence_coverage(
                source_samples,
                [score for row in source_samples for score in row["evidence_scores"]],
            ),
            "score_missing_rate": _rate(
                sum(
                    row["expected_score_slots"] - row["valid_score_slots"] for row in source_samples
                ),
                sum(row["expected_score_slots"] for row in source_samples),
            ),
            "rollout_evaluation_missing_rate": _rate(
                sum(
                    row["expected_rollout_slots"] - row["valid_rollout_slots"]
                    for row in source_samples
                ),
                sum(row["expected_rollout_slots"] for row in source_samples),
            ),
            "selection": {
                method: _metric(selection, "selection", method)["value"] for method in METHODS
            },
            "ranking": {method: _metric(ranking, "ranking", method)["value"] for method in METHODS},
        }
    return result


def _cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "expected_scoring_artifacts": 0,
        "scoring_artifacts": 0,
        "missing_scoring_artifacts": 0,
        "expected_rollout_slots": 0,
        "rollout_slots_with_usage": 0,
        "missing_rollout_usage_slots": 0,
        "known_usage_calls": 0,
        "unknown_usage_calls": 0,
        "tokens": {"input": 0, "output": 0, "reasoning": 0},
        "unknown_token_calls": {"input": 0, "output": 0, "reasoning": 0},
    }
    for row in rows:
        for route in row["route_records"]:
            if route["prefix_status"] != "available":
                continue
            summary["expected_scoring_artifacts"] += 1
            summary["expected_rollout_slots"] += 2
            cost = route.get("score_cost")
            overall = cost.get("overall") if isinstance(cost, Mapping) else None
            if isinstance(overall, Mapping):
                summary["scoring_artifacts"] += 1
                for key in ("known_usage_calls", "unknown_usage_calls"):
                    summary[key] += int(overall.get(key, 0) or 0)
                for name in summary["tokens"]:
                    summary["tokens"][name] += int((overall.get("tokens") or {}).get(name, 0) or 0)
                    summary["unknown_token_calls"][name] += int(
                        (overall.get("unknown_token_calls") or {}).get(name, 0) or 0
                    )
            usages = route.get("rollout_usages", [])
            for index in range(2):
                usage = usages[index] if index < len(usages) else None
                if not isinstance(usage, Mapping):
                    summary["missing_rollout_usage_slots"] += 1
                    summary["unknown_usage_calls"] += 1
                    continue
                summary["rollout_slots_with_usage"] += 1
                if usage.get("status") == "known":
                    summary["known_usage_calls"] += 1
                else:
                    summary["unknown_usage_calls"] += 1
                for name, key in (
                    ("input", "input_tokens"),
                    ("output", "output_tokens"),
                    ("reasoning", "reasoning_tokens"),
                ):
                    value = usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        summary["tokens"][name] += value
                    else:
                        summary["unknown_token_calls"][name] += 1
    summary["missing_scoring_artifacts"] = (
        summary["expected_scoring_artifacts"] - summary["scoring_artifacts"]
    )
    if summary["missing_scoring_artifacts"] or summary["missing_rollout_usage_slots"]:
        summary["usage_status"] = "incomplete"
    elif summary["unknown_usage_calls"]:
        summary["usage_status"] = "partial" if summary["known_usage_calls"] else "unknown"
    else:
        summary["usage_status"] = "known"
    return summary


def _write_coverage_figure(report: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    sources = report["by_source"]
    labels = list(sources)
    x = np.arange(len(labels))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for index, (key, title, color) in enumerate(
        (
            ("candidate_samples", "Candidate samples", "#708090"),
            ("included_samples", "Included samples", "#3973ac"),
            ("selection_question_groups", "Selection groups", "#52a675"),
            ("ranking_question_groups", "Ranking groups", "#8064a2"),
        )
    ):
        axes[0].bar(
            x + (index - 1.5) * width,
            [sources[label][key] for label in labels],
            width,
            label=title,
            color=color,
        )
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Coverage by source")
    axes[0].legend(fontsize="small")
    distribution = report["evidence_coverage"]["score_distribution"]
    axes[1].bar(list(distribution) or ["none"], list(distribution.values()) or [0], color="#8064a2")
    axes[1].set_xlabel("Verified Progress score")
    axes[1].set_ylabel("Routes")
    axes[1].set_title("Score coverage")
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
