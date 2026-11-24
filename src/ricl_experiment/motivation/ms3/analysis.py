"""Fixed-denominator MS3 diagnosis metrics and question-group bootstrap analysis."""

from __future__ import annotations

import itertools
import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import METHOD_IDS

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 2026
STATUS_VALUES = {
    "completed",
    "abstained",
    "parse_error",
    "invalid_output",
    "api_failure",
    "run_failure",
}
METRIC_NAMES = (
    "detection_f1",
    "first_error_accuracy",
    "false_positive_rate",
    "explicit_no_error_rate",
    "localization_coverage",
    "late_report_rate",
    "mean_signed_distance",
    "mean_absolute_distance",
)


def analyze_diagnoses(run: Mapping[str, Any] | Path, audit: Mapping[str, Any] | Path) -> dict[str, Any]:
    """Join blinded outputs to restricted labels and calculate reproducible metrics."""
    run_value = _load_object(run)
    audit_value = _load_object(audit)
    samples = _validate_samples(run_value.get("samples"))
    run_fingerprint = run_value.get("manifest_fingerprint")
    if run_fingerprint is not None and audit_value.get("manifest_fingerprint") != run_fingerprint:
        raise ValueError("run and audit manifest fingerprints do not match")
    labels = _validate_labels(audit_value.get("processbench_labels"), samples)
    paired_rows = _paired_rows(samples, labels, run_value.get("methods"))
    method_metrics = {
        method_id: _metrics(
            [
                row[method_id] | {
                    "expert_label": row["expert_label"],
                    "question_group_id": row["question_group_id"],
                }
                for row in paired_rows
            ]
        )
        for method_id in METHOD_IDS
    }
    uncertainty = _cluster_bootstrap(paired_rows, method_metrics)
    return {
        "schema_version": 1,
        "study": "ms3",
        "run_id": run_value.get("run_id"),
        "manifest_id": run_value.get("manifest_id"),
        "manifest_fingerprint": run_value.get("manifest_fingerprint"),
        "trajectory_count": len(samples),
        "question_group_count": len({sample["question_group_id"] for sample in samples}),
        "methods": method_metrics,
        "uncertainty": uncertainty,
        "paired_traces": paired_rows,
    }


def write_analysis(run: Path, audit: Path, output: Path) -> Path:
    """Write a new restricted analysis artifact without replacing prior work."""
    if output.exists():
        raise ValueError(f"analysis output already exists: {output}")
    report = analyze_diagnoses(run, audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.rename(temporary, output)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _validate_samples(value: Any) -> list[dict[str, Any]]:
    """Require stable, unique identities and one legal step count per trajectory."""
    if not isinstance(value, list) or not value:
        raise ValueError("run artifact must contain a non-empty samples list")
    samples = []
    for row in value:
        if not isinstance(row, Mapping):
            raise TypeError("run samples must be objects")
        sample_id = _text(row.get("sample_id"), "sample_id")
        trajectory_id = _text(row.get("trajectory_id"), "trajectory_id")
        group_id = _text(row.get("question_group_id"), "question_group_id")
        steps = row.get("steps")
        step_count = row.get("step_count", len(steps) if isinstance(steps, list) else None)
        if (
            isinstance(step_count, bool)
            or not isinstance(step_count, int)
            or step_count < 2
        ):
            raise ValueError(f"{sample_id}: invalid step_count")
        samples.append(
            {
                "sample_id": sample_id,
                "trajectory_id": trajectory_id,
                "question_group_id": group_id,
                "step_count": step_count,
            }
        )
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise ValueError("run sample IDs must be unique")
    if len({sample["trajectory_id"] for sample in samples}) != len(samples):
        raise ValueError("run trajectory IDs must be unique")
    return samples


def _validate_labels(value: Any, samples: list[dict[str, Any]]) -> dict[str, int]:
    """Require exactly one legal ProcessBench label for each frozen trajectory."""
    if not isinstance(value, list):
        raise TypeError("audit must contain processbench_labels")
    sample_trajectories = {sample["trajectory_id"] for sample in samples}
    labels: dict[str, int] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise TypeError("ProcessBench label rows must be objects")
        trajectory_id = _text(row.get("trajectory_id"), "trajectory_id")
        if trajectory_id not in sample_trajectories:
            continue
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label < -1:
            raise ValueError(f"{trajectory_id}: expert label must be -1 or a zero-based index")
        if trajectory_id in labels:
            raise ValueError(f"duplicate expert label for {trajectory_id}")
        labels[trajectory_id] = label
    sample_by_trajectory = {sample["trajectory_id"]: sample for sample in samples}
    if set(labels) != set(sample_by_trajectory):
        raise ValueError("run trajectories and expert-label trajectories do not match")
    for trajectory_id, label in labels.items():
        step_count = sample_by_trajectory[trajectory_id]["step_count"]
        if step_count is not None and label >= step_count:
            raise ValueError(f"{trajectory_id}: expert label is outside the trace")
    return labels


def _paired_rows(
    samples: list[dict[str, Any]], labels: dict[str, int], methods: Any
) -> list[dict[str, Any]]:
    """Join one output from every method to each blinded trace and its cluster."""
    if not isinstance(methods, Mapping) or set(methods) != set(METHOD_IDS):
        raise ValueError("run artifact must contain exactly the three MS3 methods")
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    expected = {sample["sample_id"] for sample in samples}
    for method_id in METHOD_IDS:
        rows = methods[method_id]
        if not isinstance(rows, list):
            raise TypeError(f"{method_id}: results must be a list")
        indexed[method_id] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError(f"{method_id}: result rows must be objects")
            sample_id = _text(row.get("sample_id"), "sample_id")
            if sample_id in indexed[method_id]:
                raise ValueError(f"{method_id}: duplicate result for {sample_id}")
            if sample_id not in expected:
                raise ValueError(f"{method_id}: unknown sample ID {sample_id}")
            indexed[method_id][sample_id] = row
        if set(indexed[method_id]) != expected:
            raise ValueError(f"{method_id}: every frozen sample must have one result")

    paired = []
    for sample in samples:
        label = labels[sample["trajectory_id"]]
        row: dict[str, Any] = {**sample, "expert_label": label}
        for method_id in METHOD_IDS:
            result = indexed[method_id][sample["sample_id"]]
            status = result.get("status")
            if status not in STATUS_VALUES:
                raise ValueError(f"{method_id}/{sample['sample_id']}: unknown status")
            diagnosis = result.get("diagnosis")
            has_error, position = _prediction(diagnosis, status, sample["step_count"])
            row[method_id] = {
                "status": status,
                "has_error": has_error,
                "first_error_step": position,
                "failure_classification": result.get("failure_classification"),
                "error": result.get("error"),
            }
            if label >= 0 and position is not None:
                row[method_id]["signed_distance"] = position - label
                row[method_id]["absolute_distance"] = abs(position - label)
        paired.append(row)
    return paired


def _prediction(
    diagnosis: Any, status: str, step_count: int | None
) -> tuple[bool | None, int | None]:
    """Validate a saved structured prediction before including it in metrics."""
    if status in {"parse_error", "invalid_output", "api_failure", "run_failure"}:
        if diagnosis is not None:
            raise ValueError(f"{status} result cannot contain a diagnosis")
        return None, None
    if not isinstance(diagnosis, Mapping):
        raise TypeError(f"{status} result requires a diagnosis object")
    has_error = diagnosis.get("has_error")
    position = diagnosis.get("first_error_step")
    if has_error is not True and has_error is not False and has_error is not None:
        raise ValueError("diagnosis has_error must be bool or null")
    if position is not None and (
        isinstance(position, bool)
        or not isinstance(position, int)
        or (step_count is not None and not 0 <= position < step_count)
    ):
        raise ValueError("diagnosis first_error_step is outside the trace")
    if (has_error is False or has_error is None) and position is not None:
        raise ValueError("diagnosis position contradicts has_error")
    if (status == "abstained") != (has_error is None):
        raise ValueError("diagnosis status and has_error disagree")
    return has_error, position


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate each diagnostic measure with the protocol's fixed label denominator."""
    statuses = Counter(row["status"] for row in rows)
    errors = [row for row in rows if row["expert_label"] >= 0]
    correct = [row for row in rows if row["expert_label"] == -1]
    tp = sum(row["has_error"] is True for row in errors)
    fp = sum(row["has_error"] is True for row in correct)
    fn = len(errors) - tp
    tn = sum(row["has_error"] is False for row in correct)
    exact = sum(row["first_error_step"] == row["expert_label"] for row in errors)
    localized = [row for row in errors if row["first_error_step"] is not None]
    late = sum(row["first_error_step"] > row["expert_label"] for row in localized)
    signed = [row["signed_distance"] for row in localized]
    absolute = [row["absolute_distance"] for row in localized]
    f1_denominator = 2 * tp + fp + fn
    return {
        "trajectory_count": len(rows),
        "question_group_count": len({row["question_group_id"] for row in rows}),
        "error_trajectory_denominator": len(errors),
        "correct_trajectory_denominator": len(correct),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "status_counts": {status: statuses.get(status, 0) for status in sorted(STATUS_VALUES)},
        "false_negative_status_counts": dict(
            Counter(row["status"] for row in errors if row["has_error"] is not True)
        ),
        "detection_f1": _rate(2 * tp, f1_denominator),
        "first_error_accuracy": _rate(exact, len(errors)),
        "false_positive_rate": _rate(fp, len(correct)),
        "explicit_no_error_rate": _rate(
            sum(row["has_error"] is False for row in correct), len(correct)
        ),
        "localization_coverage": _rate(len(localized), len(errors)),
        "late_report_rate": _rate(late, len(errors)),
        "localization_distance": {
            "denominator": len(localized),
            "signed": signed,
            "absolute": absolute,
            "mean_signed": float(np.mean(signed)) if signed else None,
            "mean_absolute": float(np.mean(absolute)) if absolute else None,
        },
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    """Keep numerator and frozen denominator visible, even for undefined ratios."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]], method_metrics: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Bootstrap whole original-question groups and reuse each draw across methods."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["question_group_id"]].append(row)
    group_ids = sorted(groups)
    rng = random.Random(BOOTSTRAP_SEED)
    metric_draws = {method_id: defaultdict(list) for method_id in METHOD_IDS}
    difference_draws: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for _ in range(BOOTSTRAP_REPLICATES):
        drawn_ids = [rng.choice(group_ids) for _ in group_ids]
        replicate_rows = [row for group_id in drawn_ids for row in groups[group_id]]
        replicate_metrics = {
            method_id: _metrics(
                [
                    row[method_id]
                    | {
                        "question_group_id": row["question_group_id"],
                        "expert_label": row["expert_label"],
                    }
                    for row in replicate_rows
                ]
            )
            for method_id in METHOD_IDS
        }
        for method_id, metrics in replicate_metrics.items():
            for metric in METRIC_NAMES:
                value = _metric_value(metrics, metric)
                if value is not None:
                    metric_draws[method_id][metric].append(value)
        for left, right in itertools.combinations(METHOD_IDS, 2):
            pair_id = f"{left}__minus__{right}"
            for metric in METRIC_NAMES:
                left_value = _metric_value(replicate_metrics[left], metric)
                right_value = _metric_value(replicate_metrics[right], metric)
                if left_value is not None and right_value is not None:
                    difference_draws[pair_id][metric].append(left_value - right_value)

    return {
        "method": "paired percentile cluster bootstrap",
        "cluster": "question_group_id",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "method_intervals": {
            method_id: {
                metric: _interval(metric_draws[method_id][metric])
                for metric in METRIC_NAMES
            }
            for method_id in METHOD_IDS
        },
        "paired_differences": {
            f"{left}__minus__{right}": {
                "metrics": {
                    metric: {
                        "estimate": _point_difference(method_metrics, left, right, metric),
                        **_interval(difference_draws[f"{left}__minus__{right}"][metric]),
                    }
                    for metric in METRIC_NAMES
                }
            }
            for left, right in itertools.combinations(METHOD_IDS, 2)
        },
    }


def _metric_value(metrics: Mapping[str, Any], metric: str) -> float | None:
    """Select a scalar estimate by its stable public metric name."""
    if metric == "mean_signed_distance":
        value = metrics["localization_distance"]["mean_signed"]
    elif metric == "mean_absolute_distance":
        value = metrics["localization_distance"]["mean_absolute"]
    else:
        value = metrics[metric]["value"]
    return None if value is None else float(value)


def _interval(values: list[float]) -> dict[str, Any]:
    """Return a deterministic percentile interval and the valid replicate count."""
    interval = np.quantile(values, [0.025, 0.975]).tolist() if values else None
    return {"valid_replicates": len(values), "ci_95": interval}


def _point_difference(
    metrics: dict[str, dict[str, Any]], left: str, right: str, metric: str
) -> float | None:
    """Return paired point-estimate difference when both metrics are defined."""
    left_value = _metric_value(metrics[left], metric)
    right_value = _metric_value(metrics[right], metric)
    return None if left_value is None or right_value is None else left_value - right_value


def _load_object(value: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    """Load one JSON artifact or accept its already-parsed mapping."""
    loaded = json.loads(value.read_text(encoding="utf-8")) if isinstance(value, Path) else value
    if not isinstance(loaded, Mapping):
        raise TypeError("MS3 analysis artifacts must be JSON objects")
    return loaded


def _text(value: Any, name: str) -> str:
    """Require a non-empty stable identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
