"""Deterministic offline MS2 Pilot readiness and reproducibility report."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ..analysis.ms2 import route_flow_status

FROZEN_SOURCE_COUNTS = {"math500": 5, "omni_math": 5}
METRIC_NAMES = ("route_selection_success", "within_problem_ranking_accuracy")
MINIMUM_ANALYZABLE = {"complete_topics": 8, "ranking_topics": 5, "independent_evidence_topics": 5}
_EXTERNAL = {
    "real_api_smoke": ("passed", "live"),
    "human_blind_review": ("passed", "human_blind_review"),
    "budget_signoff": ("signed", "budget_signoff"),
}
_LEAK_MARKERS = (
    "answer_key", "reference_answer", "reference_solution", "expert_label",
    "ground_truth", "gold_answer", "standard_answer", "hidden_answer",
    "rater_score", "evidence_score", "verified_progress", "future_rollout", "independent_answer",
)


class PilotArtifactError(ValueError):
    """An input artifact is not valid JSON evidence for this report."""


def build_pilot_report(
    manifest_or_run_dir: Mapping[str, Any] | Path,
    trace: Mapping[str, Any] | Path | None = None,
    *,
    requests: Iterable[Mapping[str, Any]] | Path | None = None,
    resume_evidence: Mapping[str, Any] | Path | None = None,
    scoring: Iterable[Mapping[str, Any]] | Path | None = None,
    reported_metrics: Mapping[str, Any] | Path | None = None,
    planning: Mapping[str, Any] | Path | None = None,
    external_evidence: Mapping[str, Any] | Path | None = None,
    execution_limits: Mapping[str, Any] | Path | None = None,
) -> dict[str, Any]:
    """Analyze persisted MS2 artifacts only; this function never calls a model."""
    run_dir = None
    if isinstance(manifest_or_run_dir, Path) and manifest_or_run_dir.is_dir():
        run_dir = manifest_or_run_dir
        manifest = _load_object(run_dir / "manifest.json", "manifest")
        trace = trace or run_dir / "traces" / "ms2.json"
        if resume_evidence is None:
            resume_evidence = _read_resume_ledger(run_dir / "run.sqlite3")
        if execution_limits is None:
            execution_limits = _execution_limits_from_config(run_dir / "config.json")
    else:
        manifest = _load_object(manifest_or_run_dir, "manifest")
    trace_data = _load_object(trace, "trace")
    request_rows = _load_rows(requests, "requests")
    scoring_rows = _load_rows(scoring, "scoring")
    resume_data = _optional_object(resume_evidence, "resume_evidence")
    reported_data = _optional_object(reported_metrics, "reported_metrics")
    planning_data = _optional_object(planning, "planning")
    external_data = _optional_object(external_evidence, "external_evidence")
    limits_data = _optional_object(execution_limits, "execution_limits")

    shape = _shape(manifest)
    coverage = _coverage(manifest, trace_data)
    execution = _execution_report(trace_data, limits_data)
    metrics = recompute_ms2_metrics(trace_data, scoring_rows, bootstrap_seed=_sampling_seed(manifest))
    thresholds = _thresholds(metrics)
    leakage = _leakage(request_rows)
    resume = _resume(resume_data)
    offline = _offline(metrics, reported_data)
    evidence = _evidence(scoring_rows, metrics)
    parallel = _parallel(metrics)
    precision = _precision(metrics, planning_data, shape)
    external = _external(external_data)
    checks = {
        "frozen_shape": shape["status"] == "pass",
        "checkpoint_rollout_coverage": coverage["status"] == "pass",
        "effective_limits_and_usage": execution["status"] == "pass",
        "leakage_audit": leakage["status"] == "pass",
        "interruption_resume": resume["status"] == "pass",
        "offline_recomputation": offline["status"] == "consistent",
        "minimum_analyzable": thresholds["status"] == "pass",
        "precision_planning": precision["status"] == "computed",
    }
    engineering_ready = all(checks.values())
    blockers = [
        name if item["status"] == "blocked" else f"{name}_missing"
        for name, item in external.items()
        if item["status"] != "verified"
    ]
    report = {
        "schema_version": 1,
        "study": "ms2",
        "phase": "pilot",
        "status": _status(engineering_ready, bool(blockers)),
        "engineering_ready": engineering_ready,
        "external_blocked": bool(blockers),
        "external_blockers": blockers,
        "engineering_checks": checks,
        "shape": shape,
        "coverage": coverage,
        "execution_budget": execution,
        "coverage_diagnostics": _coverage_diagnostics(metrics, coverage),
        "leakage_audit": leakage,
        "resume_evidence": resume,
        "offline_recomputation": offline,
        "thresholds": thresholds,
        "metrics": metrics,
        "evidence_diagnostics": evidence,
        "parallel_diagnostics": parallel,
        "precision_planning": precision,
        "external_evidence": external,
    }
    if run_dir is not None:
        report["run_dir"] = str(run_dir)
    return report


def build_ms2_pilot_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Named alias for downstream CLI integration."""
    return build_pilot_report(*args, **kwargs)


def write_pilot_report(manifest_or_run_dir: Mapping[str, Any] | Path, output: Path, **kwargs: Any) -> dict[str, Any]:
    """Write a reproducible JSON report atomically."""
    report = build_pilot_report(manifest_or_run_dir, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def recompute_ms2_metrics(
    trace: Mapping[str, Any] | Path,
    scoring: Iterable[Mapping[str, Any]] | Path | None = None,
    *,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Recompute frozen MS2 selection and ranking metrics without model access."""
    trace_data = _load_object(trace, "trace")
    score_map, duplicates = _score_map(_load_rows(scoring, "scoring"))
    topics: dict[str, dict[str, Any]] = {}
    for sample in _rows(trace_data.get("samples"), "trace.samples"):
        sample_id = _text(sample.get("sample_id"), "sample_id")
        candidates = []
        for candidate in _rows(sample.get("candidates"), f"candidates for {sample_id}"):
            if route_flow_status(candidate) != "eligible":
                continue
            candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
            outcome = _outcome(candidate)
            score = score_map.get(candidate_id) or score_map.get(candidate.get("route_id"), {})
            candidates.append({
                "candidate_id": candidate_id,
                "route_id": candidate.get("route_id"),
                "y": outcome["y"],
                "successes": outcome["successes"],
                "rollouts": outcome["rollouts"],
                "route_complete": outcome["complete"],
                "rater_score": _score(score, "rater_score", "rater", "llm_rater"),
                "evidence_score": _score(score, "evidence_score", "verified_progress", "progress_score"),
                "evidence_source": score.get("evidence_source"),
                "evidence_verdict": score.get("evidence_verdict"),
                "independent_evidence": _independent(score),
                "missing_reason": score.get("missing_reason", score.get("failure")),
            })
        candidates.sort(key=lambda row: row["candidate_id"])
        complete = len(candidates) >= 2 and all(row["route_complete"] for row in candidates)
        scored = complete and all(row["rater_score"] is not None and row["evidence_score"] is not None for row in candidates)
        outcomes = [row["y"] for row in candidates if row["y"] is not None]
        topics[sample_id] = {
            "question_group_id": sample.get("question_group_id"),
            "source": _source(sample.get("source")),
            "candidates": candidates,
            "route_complete": complete,
            "main_complete": scored,
            "ranking_eligible": scored and len(set(outcomes)) > 1,
            "independent_evidence": scored and any(row["independent_evidence"] for row in candidates),
        }

    selection: dict[str, list[dict[str, Any]]] = {name: [] for name in ("rater", "evidence", "combined", "random")}
    ranking: dict[str, list[dict[str, Any]]] = {name: [] for name in ("rater", "evidence", "combined", "random")}
    for sample_id, topic in sorted(topics.items()):
        if not topic["main_complete"]:
            continue
        rows = topic["candidates"]
        for method in ("rater", "evidence", "combined"):
            selected = _selection(rows, method)
            ranked = _rank_accuracy(rows, method)
            if selected is not None:
                selection[method].append({"sample_id": sample_id, "value": selected})
            if ranked is not None:
                ranking[method].append({"sample_id": sample_id, "value": ranked})
        selection["random"].append({"sample_id": sample_id, "value": mean(row["y"] for row in rows)})
        if topic["ranking_eligible"]:
            ranking["random"].append({"sample_id": sample_id, "value": 0.5})
    selection_summary = {name: _aggregate(rows) for name, rows in selection.items()}
    ranking_summary = {name: _aggregate(rows) for name, rows in ranking.items()}
    result = {
        "version": 1,
        "topics": topics,
        "duplicates": duplicates,
        "route_selection_success": selection_summary,
        "within_problem_ranking_accuracy": ranking_summary,
        "paired_observations": {
            "route_selection_success": _paired(selection, "combined", "rater"),
            "within_problem_ranking_accuracy": _paired(ranking, "combined", "rater"),
        },
    }
    result["analysis_seed"] = bootstrap_seed
    return result


def _load_object(value: Mapping[str, Any] | Path | None, name: str) -> Mapping[str, Any]:
    if value is None:
        raise PilotArtifactError(f"{name} artifact is required")
    loaded = json.loads(value.read_text(encoding="utf-8")) if isinstance(value, Path) else value
    if not isinstance(loaded, Mapping):
        raise PilotArtifactError(f"{name} artifact must be a JSON object")
    return loaded


def _optional_object(value: Mapping[str, Any] | Path | None, name: str) -> Mapping[str, Any] | None:
    return None if value is None else _load_object(value, name)


def _load_rows(value: Iterable[Mapping[str, Any]] | Path | None, name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Path) and value.suffix.casefold() == ".jsonl":
        loaded = [json.loads(line) for line in value.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        loaded = json.loads(value.read_text(encoding="utf-8")) if isinstance(value, Path) else value
    if isinstance(loaded, Mapping):
        for key in ("rows", "requests", "scoring", "records"):
            if key in loaded:
                loaded = loaded[key]
                break
    if isinstance(loaded, (str, bytes)) or not isinstance(loaded, Iterable):
        raise PilotArtifactError(f"{name} artifact must be a JSON array of objects")
    rows = list(loaded)
    if not all(isinstance(row, Mapping) for row in rows):
        raise PilotArtifactError(f"{name} rows must be JSON objects")
    return [dict(row) for row in rows]


def _rows(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise PilotArtifactError(f"{name} must be a JSON array of objects")
    return [dict(row) for row in value]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotArtifactError(f"{name} must be a non-empty string")
    return value


def _source(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold())
    if normalized == "math500":
        return "math500"
    if normalized == "omnimath":
        return "omni_math"
    return None


def _shape(manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if manifest.get("study") != "ms2":
        failures.append("study")
    if manifest.get("phase") != "pilot":
        failures.append("phase")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        return {
            "status": "fail",
            "expected": {"total": 10, **FROZEN_SOURCE_COUNTS},
            "observed": {"total": 0, "math500": 0, "omni_math": 0},
            "group_counts": {},
            "failures": [*failures, "samples"],
        }
    source_counts = Counter(_source(row.get("source")) for row in samples if isinstance(row, Mapping))
    source_counts.pop(None, None)
    observed = {"total": len(samples), "math500": source_counts.get("math500", 0), "omni_math": source_counts.get("omni_math", 0)}
    if observed != {"total": 10, **FROZEN_SOURCE_COUNTS}:
        failures.append("source_counts")
    group_counts: dict[str, int] = {}
    duplicate_groups: list[str] = []
    missing_groups = 0
    source_groups: dict[str, list[str]] = defaultdict(list)
    all_groups: list[str] = []
    sample_ids: list[str] = []
    record_ids: list[str] = []
    for row in samples:
        if not isinstance(row, Mapping):
            failures.append("sample_row")
            continue
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            sample_ids.append(sample_id)
        source = _source(row.get("source"))
        group = row.get("question_group_id")
        if source is None:
            failures.append("unknown_source")
        elif not isinstance(group, str) or not group.strip():
            missing_groups += 1
        else:
            source_groups[source].append(group)
            all_groups.append(group)
        record_id = row.get("source_record_id")
        if isinstance(record_id, str) and record_id:
            record_ids.append(record_id)
    for source, groups in source_groups.items():
        group_counts[source] = len(set(groups))
        duplicate_groups.extend(f"{source}:{group}" for group, count in Counter(groups).items() if count > 1)
        if len(set(groups)) != FROZEN_SOURCE_COUNTS.get(source):
            failures.append(f"group_count:{source}")
    if missing_groups:
        failures.append("missing_question_group_id")
    duplicate_global_groups = sorted(group for group, count in Counter(all_groups).items() if count > 1)
    if len(set(sample_ids)) != len(sample_ids):
        failures.append("duplicate_sample_id")
    if len(record_ids) != len(set(record_ids)):
        failures.append("duplicate_source_record_id")
    return {
        "status": "pass" if not failures else "fail",
        "expected": {"total": 10, **FROZEN_SOURCE_COUNTS},
        "observed": observed,
        "group_counts": group_counts,
        "duplicate_groups": sorted(duplicate_groups),
        "duplicate_global_question_groups": duplicate_global_groups,
        "missing_group_ids": missing_groups,
        "failures": sorted(set(failures)),
    }


def _coverage(manifest: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    manifest_rows = _rows(manifest.get("samples"), "manifest.samples") if isinstance(manifest.get("samples"), list) else []
    trace_rows = _rows(trace.get("samples"), "trace.samples") if isinstance(trace.get("samples"), list) else []
    expected_candidates = len(manifest_rows) * 3
    expected_rollouts = expected_candidates * 2
    observed_candidates = 0
    available = 0
    dispatched = 0
    completed = 0
    valid = 0
    manifest_ids = {row.get("sample_id") for row in manifest_rows}
    trace_ids = {row.get("sample_id") for row in trace_rows}
    for sample in trace_rows:
        candidates = sample.get("candidates")
        if not isinstance(candidates, list):
            continue
        observed_candidates += len(candidates)
        for candidate in candidates:
            if candidate.get("prefix_status") == "available" and str(candidate.get("prefix_text", "")).strip():
                available += 1
            rollouts = candidate.get("rollouts")
            if not isinstance(rollouts, list):
                continue
            for rollout in rollouts:
                if rollout.get("status") != "not_dispatched":
                    dispatched += 1
                if rollout.get("status") == "completed":
                    completed += 1
                    if _rollout_status(rollout) in {"correct", "incorrect"}:
                        valid += 1
    failures = []
    per_sample_failures = []
    for sample in trace_rows:
        candidates = sample.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 3:
            per_sample_failures.append({"sample_id": sample.get("sample_id"), "reason": "candidate_count"})
            continue
        candidate_indices = {candidate.get("candidate_index") for candidate in candidates if isinstance(candidate, Mapping)}
        candidate_ids = [candidate.get("candidate_id") for candidate in candidates if isinstance(candidate, Mapping)]
        if candidate_indices != {0, 1, 2} or len(candidate_ids) != len(set(candidate_ids)):
            per_sample_failures.append({"sample_id": sample.get("sample_id"), "reason": "candidate_slots"})
        for candidate in candidates:
            rollouts = candidate.get("rollouts")
            indices = {
                rollout.get("rollout_index")
                for rollout in rollouts
                if isinstance(rollout, Mapping)
            } if isinstance(rollouts, list) else set()
            if indices != {0, 1} or len(rollouts) != 2:
                per_sample_failures.append({
                    "sample_id": sample.get("sample_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": "rollout_slots",
                })
    if trace.get("study") != "ms2" or trace.get("phase") != "pilot":
        failures.append("trace_identity")
    if trace.get("candidate_slots") != 3 or trace.get("rollouts_per_checkpoint") != 2:
        failures.append("frozen_trace_parameters")
    if manifest_ids != trace_ids:
        failures.append("sample_identity")
    if observed_candidates != expected_candidates:
        failures.append("checkpoint_count")
    if available != expected_candidates:
        failures.append("checkpoint_availability")
    if completed != expected_rollouts:
        failures.append("rollout_completion")
    if per_sample_failures:
        failures.append("per_sample_slots")
    return {
        "status": "pass" if not failures else "fail",
        "checkpoints": {"expected": expected_candidates, "observed": observed_candidates, "available": available, "coverage": _ratio(available, expected_candidates)},
        "rollouts": {"expected": expected_rollouts, "dispatched": dispatched, "completed": completed, "valid_evaluations": valid, "coverage": _ratio(completed, expected_rollouts), "valid_coverage": _ratio(valid, expected_rollouts)},
        "missing_sample_ids": sorted(str(value) for value in manifest_ids - trace_ids if value),
        "per_sample_failures": per_sample_failures,
        "failures": failures,
    }


def _outcome(candidate: Mapping[str, Any]) -> dict[str, Any]:
    rollouts = candidate.get("rollouts")
    if not isinstance(rollouts, list):
        return {"complete": False, "rollouts": 0, "successes": None, "y": None}
    by_index = {row.get("rollout_index"): row for row in rollouts if isinstance(row, Mapping)}
    if set(by_index) != {0, 1}:
        return {"complete": False, "rollouts": len(rollouts), "successes": None, "y": None}
    statuses = [_rollout_status(by_index[index]) for index in (0, 1)]
    if any(by_index[index].get("status") != "completed" for index in (0, 1)) or any(status not in {"correct", "incorrect"} for status in statuses):
        return {"complete": False, "rollouts": len(rollouts), "successes": None, "y": None}
    successes = statuses.count("correct")
    return {"complete": True, "rollouts": len(rollouts), "successes": successes, "y": successes / 2}


def _rollout_status(rollout: Mapping[str, Any]) -> str | None:
    evaluation = rollout.get("evaluation")
    status = evaluation.get("status") if isinstance(evaluation, Mapping) else None
    return status if isinstance(status, str) else None


def _score_map(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = row.get("candidate_id") or row.get("route_id")
        if not isinstance(key, str) or not key:
            continue
        if key in result:
            duplicates.append(key)
        result[key] = dict(row)
    return result, sorted(set(duplicates))


def _score(row: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, Mapping):
            value = value.get("score")
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3:
            return value
    return None


def _independent(row: Mapping[str, Any]) -> bool:
    if row.get("independent_evidence") is True or row.get("evidence_eligible") is True:
        return True
    progress = row.get("verified_progress")
    if isinstance(progress, Mapping) and _score(row, "verified_progress") is not None:
        numerator = progress.get("coverage_numerator")
        return type(numerator) is int and numerator > 0
    source = str(row.get("evidence_source", "")).casefold()
    verdict = str(row.get("evidence_verdict", row.get("evidence_status", ""))).casefold()
    return source in {"tool-verified", "derivation-checked", "agent-assessed"} and verdict in {"confirmed", "verified", "falsified", "contradicted"}


def _selection(rows: Sequence[Mapping[str, Any]], method: str) -> float | None:
    values = _method_scores(rows, method)
    if values is None:
        return None
    best = max(values.values())
    return mean(row["y"] for row in rows if values[row["candidate_id"]] == best)


def _rank_accuracy(rows: Sequence[Mapping[str, Any]], method: str) -> float | None:
    values = _method_scores(rows, method)
    if values is None:
        return None
    outcomes = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left["y"] == right["y"]:
                continue
            left_observed = left["y"] > right["y"]
            left_predicted = values[left["candidate_id"]] > values[right["candidate_id"]]
            predicted_tie = values[left["candidate_id"]] == values[right["candidate_id"]]
            outcomes.append(0.5 if predicted_tie else float(left_predicted == left_observed))
    return mean(outcomes) if outcomes else None


def _method_scores(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, float] | None:
    if method == "rater":
        values = [row["rater_score"] for row in rows]
        return None if any(value is None for value in values) else {row["candidate_id"]: float(row["rater_score"]) for row in rows}
    if method == "evidence":
        values = [row["evidence_score"] for row in rows]
        return None if any(value is None for value in values) else {row["candidate_id"]: float(row["evidence_score"]) for row in rows}
    if method == "combined":
        if any(row["rater_score"] is None or row["evidence_score"] is None for row in rows):
            return None
        rater = _average_ranks([row["rater_score"] for row in rows])
        evidence = _average_ranks([row["evidence_score"] for row in rows])
        return {row["candidate_id"]: -(rater[index] + evidence[index]) / 2 for index, row in enumerate(rows)}
    return None


def _average_ranks(values: Sequence[int]) -> dict[int, float]:
    ordered = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))
    result: dict[int, float] = {}
    position = 1
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (position + position + end - index - 1) / 2
        for item_index in range(index, end):
            result[ordered[item_index][0]] = rank
        position += end - index
        index = end
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["value"]) for row in rows]
    return {"value": mean(values) if values else None, "n_topics": len(values), "values": [round(value, 12) for value in values]}


def _paired(methods: Mapping[str, Sequence[Mapping[str, Any]]], left: str, right: str) -> list[dict[str, Any]]:
    left_map = {row["sample_id"]: row["value"] for row in methods[left]}
    right_map = {row["sample_id"]: row["value"] for row in methods[right]}
    return [{"sample_id": key, "difference": left_map[key] - right_map[key]} for key in sorted(left_map.keys() & right_map.keys())]


def _thresholds(metrics: Mapping[str, Any]) -> dict[str, Any]:
    topics = metrics.get("topics", {})
    counts = {
        "complete_topics": sum(bool(topic.get("main_complete")) for topic in topics.values()),
        "ranking_topics": sum(bool(topic.get("ranking_eligible")) for topic in topics.values()),
        "independent_evidence_topics": sum(bool(topic.get("independent_evidence")) for topic in topics.values()),
    }
    rows = {name: {"observed": count, "required": MINIMUM_ANALYZABLE[name], "denominator": 10, "pass": count >= MINIMUM_ANALYZABLE[name]} for name, count in counts.items()}
    return {"status": "pass" if all(row["pass"] for row in rows.values()) else "fail", **rows}


def _coverage_diagnostics(metrics: Mapping[str, Any], coverage: Mapping[str, Any]) -> dict[str, Any]:
    topics = list(metrics.get("topics", {}).values())
    return {
        "topics": len(topics),
        "route_complete_topics": sum(bool(topic.get("route_complete")) for topic in topics),
        "main_complete_topics": sum(bool(topic.get("main_complete")) for topic in topics),
        "ranking_eligible_topics": sum(bool(topic.get("ranking_eligible")) for topic in topics),
        "independent_evidence_topics": sum(bool(topic.get("independent_evidence")) for topic in topics),
        "checkpoint_coverage": coverage["checkpoints"]["coverage"],
        "rollout_coverage": coverage["rollouts"]["coverage"],
        "valid_rollout_coverage": coverage["rollouts"]["valid_coverage"],
    }


def _evidence(rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
    valid_scores = sum(_score(row, "evidence_score", "verified_progress", "progress_score") is not None for row in rows)
    source_counts = Counter(str(row.get("evidence_source", "missing")) for row in rows)
    verdict_counts = Counter(str(row.get("evidence_verdict", "missing")) for row in rows)
    independent_candidates = sum(_independent(row) for row in rows)
    independent_topics = sum(bool(topic.get("independent_evidence")) for topic in metrics.get("topics", {}).values())
    return {
        "status": "pass" if independent_topics >= MINIMUM_ANALYZABLE["independent_evidence_topics"] else "insufficient",
        "rows": len(rows),
        "valid_scores": valid_scores,
        "missing_scores": len(rows) - valid_scores,
        "source_counts": dict(sorted(source_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "independent_evidence_candidates": independent_candidates,
        "independent_evidence_topics": independent_topics,
    }


def _parallel(metrics: Mapping[str, Any]) -> dict[str, Any]:
    tie_pairs = topics_with_ties = all_tied = 0
    for topic in metrics.get("topics", {}).values():
        if not topic.get("main_complete"):
            continue
        scores = [row["evidence_score"] for row in topic["candidates"]]
        pairs = sum(left == right for index, left in enumerate(scores) for right in scores[index + 1 :])
        tie_pairs += pairs
        topics_with_ties += bool(pairs)
        all_tied += len(set(scores)) == 1
    return {"status": "reported", "evidence_tie_pairs": tie_pairs, "topics_with_evidence_ties": topics_with_ties, "topics_with_all_evidence_scores_tied": all_tied}


def _leakage(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not requests:
        return {"status": "missing", "request_count": 0, "hits": []}
    hits = []
    for index, request in enumerate(requests):
        text = json.dumps(request, ensure_ascii=False, sort_keys=True).casefold()
        normalized = re.sub(r"[^a-z0-9]+", "_", text)
        found = sorted(marker for marker in _LEAK_MARKERS if marker in normalized)
        if found:
            hits.append({"request_index": index, "markers": found})
    return {"status": "pass" if not hits else "fail", "request_count": len(requests), "hits": hits, "forbidden_markers": list(_LEAK_MARKERS)}


def _resume(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {"status": "missing", "recovered_calls": 0, "duplicate_analysis_results": 0}
    calls = evidence.get("logical_calls", evidence.get("calls", []))
    if not isinstance(calls, list):
        return {"status": "fail", "reason": "logical_calls must be a list"}
    recovered = unresolved = duplicates = 0
    for call in calls:
        if not isinstance(call, Mapping):
            unresolved += 1
            continue
        attempts = call.get("attempts", [])
        statuses = [row.get("status") for row in attempts if isinstance(row, Mapping)] if isinstance(attempts, list) else []
        if any(status in {"unknown", "pending"} for status in statuses) and "completed" in statuses:
            recovered += 1
        if statuses and statuses[-1] not in {"completed", "failed"}:
            unresolved += 1
        results = call.get("analysis_results")
        count = len(results) if isinstance(results, list) else call.get("analysis_result_count", 1)
        if isinstance(count, int) and count > 1:
            duplicates += count - 1
    return {"status": "pass" if recovered > 0 and unresolved == 0 and duplicates == 0 else "fail", "recovered_calls": recovered, "unresolved_calls": unresolved, "duplicate_analysis_results": duplicates}


def _read_resume_ledger(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            calls = []
            for call in db.execute("SELECT logical_call_id FROM logical_calls ORDER BY logical_call_id"):
                call_id = call["logical_call_id"]
                attempts = [dict(row) for row in db.execute("SELECT attempt_number, status FROM attempts WHERE logical_call_id = ? ORDER BY attempt_number", (call_id,))]
                count = db.execute("SELECT COUNT(*) FROM analysis_results WHERE logical_call_id = ?", (call_id,)).fetchone()[0]
                calls.append({"logical_call_id": call_id, "attempts": attempts, "analysis_result_count": count})
        return {"logical_calls": calls}
    except sqlite3.Error:
        return None



def _execution_limits_from_config(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    budget = payload.get("budget") if isinstance(payload, Mapping) else None
    return dict(budget) if isinstance(budget, Mapping) else None


def _execution_report(trace: Mapping[str, Any], limits: Mapping[str, Any] | None) -> dict[str, Any]:
    if limits is None:
        return {"status": "missing", "reason": "effective attempt and output-token limits were not supplied"}
    max_attempts = limits.get("max_attempts_per_call")
    max_reserved = limits.get("max_reserved_output_tokens")
    usage = trace.get("usage") if isinstance(trace.get("usage"), Mapping) else {}
    known = usage.get("known_output_tokens", limits.get("known_output_tokens", 0))
    reserved = usage.get("reserved_output_tokens", limits.get("reserved_output_tokens", 0))
    failures = []
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        failures.append("max_attempts_per_call")
    if not isinstance(max_reserved, int) or isinstance(max_reserved, bool) or max_reserved < 1:
        failures.append("max_reserved_output_tokens")
    if not _number(known) or known < 0 or not _number(reserved) or reserved < 0:
        failures.append("usage")
    elif isinstance(max_reserved, int) and known + reserved > max_reserved:
        failures.append("reserved_output_budget_exceeded")
    observed_attempts = limits.get("observed_attempts")
    if observed_attempts is not None and (not isinstance(observed_attempts, int) or observed_attempts < 0):
        failures.append("observed_attempts")
    return {
        "status": "pass" if not failures else "fail",
        "max_attempts_per_call": max_attempts,
        "max_reserved_output_tokens": max_reserved,
        "observed_attempts": observed_attempts,
        "known_output_tokens": known,
        "reserved_output_tokens": reserved,
        "failures": failures,
    }


def _sampling_seed(manifest: Mapping[str, Any]) -> int:
    sampling = manifest.get("sampling")
    seed = sampling.get("seed") if isinstance(sampling, Mapping) else None
    return seed if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 else 0


def _offline(computed: Mapping[str, Any], reported: Mapping[str, Any] | None) -> dict[str, Any]:
    view = _metric_view(computed)
    if reported is None:
        return {"status": "not_checked", "computed_digest": _digest(view)}
    reported_view = _metric_view(reported.get("metrics", reported) if isinstance(reported, Mapping) else reported)
    mismatches = _differences(view, reported_view)
    result = {"status": "consistent" if not mismatches else "mismatch", "computed_digest": _digest(view), "reported_digest": _digest(reported_view), "mismatches": mismatches}
    return result


def _metric_view(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"route_selection_success": metrics.get("route_selection_success"), "within_problem_ranking_accuracy": metrics.get("within_problem_ranking_accuracy"), "paired_observations": metrics.get("paired_observations")}


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "root"]
    if isinstance(left, Mapping):
        result = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                result.append(f"{path}.{key}".lstrip("."))
            else:
                result.extend(_differences(left[key], right[key], f"{path}.{key}".lstrip(".")))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        result = []
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            result.extend(_differences(left_value, right_value, f"{path}[{index}]"))
        return result
    return [] if left == right else [path]


def _precision(metrics: Mapping[str, Any], planning: Mapping[str, Any] | None, shape: Mapping[str, Any]) -> dict[str, Any]:
    if planning is None:
        return {"status": "missing_artifact", "reason": "precision and cost inputs were not supplied"}
    targets = planning.get("targets")
    rates = planning.get("cost_rates")
    usage = planning.get("usage")
    conservative = planning.get("conservative", {})
    if not all(isinstance(value, Mapping) for value in (targets, rates, usage, conservative)):
        return {"status": "invalid_artifact", "reason": "targets, cost_rates, usage, and conservative are required"}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    input_rate = rates.get("input_per_1k")
    output_rate = rates.get("output_per_1k")
    if not all(_number(value) and value >= 0 for value in (input_tokens, output_tokens, input_rate, output_rate)):
        return {"status": "invalid_artifact", "reason": "non-negative token usage and cost rates are required"}
    pilot_topics = usage.get("pilot_topics", shape.get("observed", {}).get("total", 0))
    if not isinstance(pilot_topics, int) or pilot_topics <= 0:
        return {"status": "invalid_artifact", "reason": "usage.pilot_topics must be positive"}
    total_cost = input_tokens / 1000 * input_rate + output_tokens / 1000 * output_rate
    cost_per_topic = total_cost / pilot_topics
    sd_multiplier = conservative.get("sd_multiplier")
    rate_multiplier = conservative.get("valid_rate_multiplier")
    if not _number(sd_multiplier) or sd_multiplier <= 0 or not _number(rate_multiplier) or rate_multiplier <= 0:
        return {"status": "invalid_artifact", "reason": "conservative multipliers must be positive"}
    plans = {}
    for metric in METRIC_NAMES:
        target = targets.get(metric)
        if not isinstance(target, Mapping):
            plans[metric] = {"status": "missing_target"}
            continue
        half_width = target.get("target_half_width")
        meaningful = target.get("minimum_meaningful_difference")
        if not _number(half_width) or half_width <= 0 or not _number(meaningful) or meaningful <= 0:
            plans[metric] = {"status": "invalid_target"}
            continue
        values = [float(row["difference"]) for row in metrics.get("paired_observations", {}).get(metric, []) if _number(row.get("difference"))]
        if len(values) < 2:
            plans[metric] = {"status": "insufficient_variance", "pilot_groups": len(values)}
            continue
        pilot_sd = stdev(values)
        denominator = len(metrics.get("topics", {})) or 1
        valid_rate = len(values) / denominator
        conservative_sd = pilot_sd * sd_multiplier
        conservative_rate = min(1.0, valid_rate * rate_multiplier)
        base_effective = max(1, math.ceil((1.96 * pilot_sd / half_width) ** 2))
        conservative_effective = max(1, math.ceil((1.96 * conservative_sd / half_width) ** 2))
        base_topics = max(1, math.ceil(base_effective / valid_rate))
        conservative_topics = max(1, math.ceil(conservative_effective / conservative_rate))
        plans[metric] = {
            "status": "computed",
            "pilot_groups": len(values),
            "minimum_meaningful_difference": meaningful,
            "target_half_width": half_width,
            "base": {
                "sd": pilot_sd,
                "valid_rate": valid_rate,
                "required_effective_groups": base_effective,
                "projected_topics": base_topics,
                "projected_cost": base_topics * cost_per_topic,
            },
            "conservative": {
                "sd": conservative_sd,
                "valid_rate": conservative_rate,
                "required_effective_groups": conservative_effective,
                "projected_topics": conservative_topics,
                "projected_cost": conservative_topics * cost_per_topic,
            },
            "pilot_sd": pilot_sd,
            "conservative_sd": conservative_sd,
            "valid_rate": valid_rate,
            "conservative_valid_rate": conservative_rate,
            "required_effective_groups": conservative_effective,
            "projected_topics": conservative_topics,
            "projected_cost": conservative_topics * cost_per_topic,
        }
    result = {
        "status": "computed" if all(plan.get("status") == "computed" for plan in plans.values()) else "incomplete",
        "confidence_level": 0.95,
        "method": "normal approximation from supplied pilot group differences",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "pilot_topics": pilot_topics, "pilot_cost": total_cost, "cost_per_topic": cost_per_topic},
        "plans": plans,
    }
    if isinstance(planning.get("max_budget"), (int, float)) and not isinstance(planning.get("max_budget"), bool):
        result["max_budget"] = planning["max_budget"]
        result["budget_check"] = {metric: plan.get("projected_cost") is not None and plan["projected_cost"] <= planning["max_budget"] for metric, plan in plans.items()}
    return result


def _external(evidence: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    result = {}
    for name, (status, mode) in _EXTERNAL.items():
        item = evidence.get(name) if evidence else None
        if not isinstance(item, Mapping):
            result[name] = {"status": "missing", "required": {"status": status, "mode": mode}}
        elif item.get("status") == status and item.get("mode") == mode and item.get("evidence_ref"):
            result[name] = {"status": "verified", "evidence_ref": item["evidence_ref"]}
        else:
            result[name] = {"status": "blocked", "required": {"status": status, "mode": mode}}
    return result


def _status(engineering_ready: bool, external_blocked: bool) -> str:
    if engineering_ready and external_blocked:
        return "engineering_ready_externally_blocked"
    if engineering_ready:
        return "ready_for_external_signoff"
    if external_blocked:
        return "engineering_blocked_externally_blocked"
    return "engineering_blocked"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
