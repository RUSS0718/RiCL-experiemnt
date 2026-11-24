"""Offline MS3 Pilot acceptance checks and planning artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import canonical_json
from ..motivation.ms3.analysis import analyze_diagnoses
from ..motivation.ms3.protocol import METHOD_IDS
from ..motivation.ms3_repair_protocol import GROUPS
from .ms3_pilot_support import (
    _build_planning,
    _evidence_status,
    _expected_call_ids,
    _ledger_snapshot,
    _ledger_summary,
    _load_jsonl,
    _load_object,
    _load_rows,
    _nested_keys,
    _private_text_values,
    _safe_check,
    _write_json,
)
from .ms3_repair import analyze_repairs

MIN_LEGAL_DIAGNOSES = 9
MIN_EVALUABLE_REPAIRS = 6
DEFAULT_TARGET_HALF_WIDTH = 0.25
_FORBIDDEN_REQUEST_KEYS = {
    "expert_label",
    "final_answer_correct",
    "future_outcome",
    "group",
    "label",
    "other_results",
    "position_source",
    "reference_answer",
    "reference_material",
    "reference_solution",
    "first_error_position",
}


def write_ms3_pilot_artifacts(
    manifest: Mapping[str, Any] | Path,
    audit: Mapping[str, Any] | Path,
    answer_key: Sequence[Mapping[str, Any]] | Path,
    diagnosis_run: Path,
    diagnosis_analysis: Mapping[str, Any] | Path,
    repair_run: Path,
    repair_analysis: Mapping[str, Any] | Path,
    output: Path,
    *,
    plan_output: Path | None = None,
    pricing: Mapping[str, float] | None = None,
    target_half_width: float = DEFAULT_TARGET_HALF_WIDTH,
    restart_record: Mapping[str, Any] | Path | None = None,
    real_api_record: Mapping[str, Any] | Path | None = None,
    human_signoff_record: Mapping[str, Any] | Path | None = None,
) -> dict[str, Path]:
    """Write a deterministic local engineering status and sample-size/cost plan."""
    manifest_value = _load_object(manifest)
    audit_value = _load_object(audit)
    answer_key_value = _load_rows(answer_key)
    diagnosis_analysis_value = _load_object(diagnosis_analysis)
    repair_analysis_value = _load_object(repair_analysis)
    shape = _sample_shape(manifest_value, audit_value, answer_key_value)
    diagnosis = _safe_check(
        "diagnosis_methods",
        lambda: _diagnosis_check(diagnosis_run, diagnosis_analysis_value, shape),
    )
    repair = _safe_check(
        "repair_coverage",
        lambda: _repair_check(repair_run, repair_analysis_value, shape),
    )
    leakage = _safe_check(
        "leakage",
        lambda: _leakage_check(
            diagnosis_run, repair_run, answer_key_value, shape
        ),
    )
    ledger = _safe_check(
        "ledger",
        lambda: _ledger_check(diagnosis_run, repair_run),
    )
    restart = _safe_check(
        "restart",
        lambda: _restart_check(diagnosis_run, repair_run, restart_record),
    )
    recompute = _safe_check(
        "offline_recompute",
        lambda: _offline_recompute(
            audit_value,
            diagnosis_run,
            diagnosis_analysis_value,
            repair_run,
            repair_analysis_value,
        ),
    )
    checks = {
        "sample_shape": shape,
        "diagnosis_methods": diagnosis,
        "repair_coverage": repair,
        "leakage": leakage,
        "ledger": ledger,
        "restart": restart,
        "offline_recompute": recompute,
    }
    failures = [name for name, check in checks.items() if check["status"] != "passed"]
    external_evidence = {
        "real_api": _evidence_status(real_api_record, "real_api_record"),
        "human_signoff": _evidence_status(human_signoff_record, "human_signoff_record"),
    }
    report = {
        "schema_version": 1,
        "study": "ms3",
        "phase": manifest_value.get("phase"),
        "manifest_id": manifest_value.get("manifest_id"),
        "manifest_fingerprint": manifest_value.get("fingerprint"),
        "status": "engineering_gate_passed" if not failures else "engineering_gate_failed",
        "failed_checks": failures,
        "checks": checks,
        "external_evidence": external_evidence,
        "confirmatory_readiness": _confirmatory_readiness(external_evidence),
        "scope_note": (
            "This artifact reports local engineering evidence only; it does not claim a real API "
            "pilot, credentials, or human sign-off."
        ),
    }
    plan = _build_planning(
        repair_analysis_value,
        diagnosis_run,
        repair_run,
        pricing,
        target_half_width,
    )
    output = Path(output)
    plan_output = plan_output or output.with_name("sample-size-cost.json")
    _write_json(output, report)
    _write_json(plan_output, plan)
    return {"status": output, "plan": plan_output}


def _confirmatory_readiness(external_evidence: Mapping[str, Any]) -> str:
    """Describe why a local engineering pass is not a confirmatory sign-off."""
    real_api = external_evidence["real_api"].get("status")
    signoff = external_evidence["human_signoff"].get("status")
    if real_api == "provided" and signoff == "provided":
        return "external_records_present_manual_review_required"
    if real_api == "provided":
        return "not_ready_without_human_signoff"
    return "not_ready_without_external_evidence"


def _sample_shape(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    answer_key: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify the frozen ten-trajectory, eight-error, two-no-error pilot shape."""
    errors: list[str] = []
    samples = manifest.get("samples")
    if manifest.get("study") != "ms3" or manifest.get("phase") != "pilot":
        errors.append("manifest must be an MS3 pilot manifest")
    if not isinstance(samples, list):
        errors.append("manifest samples must be a list")
        samples = []
    sample_ids = [row.get("sample_id") for row in samples if isinstance(row, Mapping)]
    groups = [row.get("question_group_id") for row in samples if isinstance(row, Mapping)]
    trajectories = [row.get("trajectory_id") for row in samples if isinstance(row, Mapping)]
    if len(samples) != 10:
        errors.append(f"expected 10 samples, observed {len(samples)}")
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("sample IDs are not unique")
    if len(set(groups)) != 10:
        errors.append("question groups are not ten independent groups")
    if len(set(trajectories)) != len(trajectories):
        errors.append("trajectory IDs are not unique")
    labels = _labels_for_samples(audit, trajectories, errors)
    answer_rows = {str(row.get("sample_id")): row for row in answer_key}
    if set(answer_rows) != {str(value) for value in sample_ids}:
        errors.append("answer key does not cover exactly the frozen samples")
    for sample_id, trajectory_id in zip(sample_ids, trajectories, strict=True):
        answer = answer_rows.get(str(sample_id))
        if answer is not None and answer.get("expert_label") != labels.get(str(trajectory_id)):
            errors.append(f"answer-key label disagrees for {sample_id}")
    error_ids = [str(sample_id) for sample_id, trajectory in zip(sample_ids, trajectories, strict=True)
                 if labels.get(str(trajectory), -2) >= 0]
    no_error_ids = [str(sample_id) for sample_id, trajectory in zip(sample_ids, trajectories, strict=True)
                    if labels.get(str(trajectory), -2) == -1]
    if len(error_ids) != 8 or len(no_error_ids) != 2:
        errors.append(f"expected 8 error and 2 no-error trajectories, got {len(error_ids)} and {len(no_error_ids)}")
    pools = audit.get("study_pools")
    if not isinstance(pools, Mapping):
        errors.append("audit is missing study pools")
    else:
        if {str(value) for value in pools.get("diagnosis", [])} != {str(value) for value in sample_ids}:
            errors.append("diagnosis pool does not equal the frozen ten-sample set")
        if {str(value) for value in pools.get("repair", [])} != set(error_ids):
            errors.append("repair pool does not equal the eight error trajectories")
    return {
        "status": "passed" if not errors else "failed",
        "required": {
            "trajectory_count": 10,
            "question_group_count": 10,
            "error_trajectory_count": 8,
            "no_error_trajectory_count": 2,
        },
        "observed": {
            "trajectory_count": len(samples),
            "question_group_count": len(set(groups)),
            "error_trajectory_count": len(error_ids),
            "no_error_trajectory_count": len(no_error_ids),
        },
        "error_sample_ids": error_ids,
        "no_error_sample_ids": no_error_ids,
        "errors": errors,
    }


def _labels_for_samples(
    audit: Mapping[str, Any], trajectories: list[Any], errors: list[str]
) -> dict[str, int]:
    """Index and validate the restricted ProcessBench labels used by the gate."""
    labels: dict[str, int] = {}
    rows = audit.get("processbench_labels")
    if not isinstance(rows, list):
        errors.append("audit is missing ProcessBench labels")
        return labels
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("ProcessBench labels contain a non-object row")
            continue
        trajectory_id = row.get("trajectory_id")
        label = row.get("label")
        if trajectory_id in labels:
            errors.append(f"duplicate ProcessBench label for {trajectory_id}")
        if isinstance(label, bool) or not isinstance(label, int) or label < -1:
            errors.append(f"invalid ProcessBench label for {trajectory_id}")
            continue
        labels[str(trajectory_id)] = label
    missing = {str(value) for value in trajectories} - set(labels)
    if missing:
        errors.append("missing ProcessBench labels: " + ", ".join(sorted(missing)))
    return labels


def _diagnosis_check(
    run_dir: Path, analysis: Mapping[str, Any], shape: Mapping[str, Any]
) -> dict[str, Any]:
    """Check all three methods have ten rows and at least nine legal outputs."""
    run = _load_object(Path(run_dir) / "run.json")
    methods = run.get("methods")
    metrics = analysis.get("methods")
    errors: list[str] = []
    details: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        rows = methods.get(method_id) if isinstance(methods, Mapping) else None
        metric = metrics.get(method_id) if isinstance(metrics, Mapping) else None
        statuses = metric.get("status_counts", {}) if isinstance(metric, Mapping) else {}
        legal = int(statuses.get("completed", 0)) + int(statuses.get("abstained", 0))
        row_count = len(rows) if isinstance(rows, list) else 0
        details[method_id] = {
            "row_count": row_count,
            "legal_output_count": legal,
            "status_counts": dict(statuses) if isinstance(statuses, Mapping) else {},
            "threshold": MIN_LEGAL_DIAGNOSES,
        }
        if row_count != shape["observed"]["trajectory_count"]:
            errors.append(f"{method_id}: expected one row per frozen trajectory")
        if isinstance(rows, list) and any(row.get("method") != method_id for row in rows):
            errors.append(f"{method_id}: result method IDs do not match their bucket")
        if legal < MIN_LEGAL_DIAGNOSES:
            errors.append(f"{method_id}: only {legal} legal outputs")
    if set(methods) != set(METHOD_IDS) if isinstance(methods, Mapping) else True:
        errors.append("diagnosis artifact does not contain exactly the three methods")
    return {"status": "passed" if not errors else "failed", "methods": details, "errors": errors}


def _repair_check(
    run_dir: Path, analysis: Mapping[str, Any], shape: Mapping[str, Any]
) -> dict[str, Any]:
    """Check one isolated G0-G3 assignment for every eligible error sample."""
    assignments = _load_rows(Path(run_dir) / "restricted" / "assignments.json")
    evaluations = _load_jsonl(Path(run_dir) / "blind" / "evaluations.jsonl")
    expected = set(shape["error_sample_ids"])
    errors: list[str] = []
    by_sample: dict[str, set[str]] = {}
    assignment_counts: dict[tuple[str, str], int] = {}
    evaluation_by_id: dict[str, Mapping[str, Any]] = {}
    for row in assignments:
        sample_id = str(row.get("sample_id"))
        group = str(row.get("group"))
        if group not in GROUPS:
            errors.append(f"unknown repair group {group}")
        by_sample.setdefault(sample_id, set()).add(group)
        assignment_key = (sample_id, group)
        assignment_counts[assignment_key] = assignment_counts.get(assignment_key, 0) + 1
    for row in evaluations:
        evaluation_id = str(row.get("evaluation_id"))
        if evaluation_id in evaluation_by_id:
            errors.append(f"duplicate evaluation ID {evaluation_id}")
        evaluation_by_id[evaluation_id] = row
    duplicate_assignments = [
        f"{sample_id}:{group}"
        for (sample_id, group), count in assignment_counts.items()
        if count != 1
    ]
    if duplicate_assignments:
        errors.append("repair assignments are not one-per-sample/group: " + ", ".join(sorted(duplicate_assignments)))
    if set(by_sample) != expected:
        errors.append("repair assignments do not cover exactly the eight error samples")
    incomplete = [sample_id for sample_id, groups in by_sample.items() if groups != set(GROUPS)]
    if incomplete:
        errors.append("incomplete G0-G3 coverage: " + ", ".join(sorted(incomplete)))
    evaluable = sum(
        set(by_sample.get(sample_id, ())) == set(GROUPS)
        and all(
            evaluation_by_id.get(str(row.get("evaluation_id")), {}).get("repair_success")
            in {"correct", "incorrect"}
            for row in assignments
            if str(row.get("sample_id")) == sample_id
        )
        for sample_id in expected
    )
    common = analysis.get("four_group_common", {})
    if common.get("n_samples") != evaluable:
        errors.append("repair analysis four-group count disagrees with raw evaluations")
    if evaluable < MIN_EVALUABLE_REPAIRS:
        errors.append(f"only {evaluable} four-group repair trajectories are independently evaluable")
    return {
        "status": "passed" if not errors else "failed",
        "eligible_error_trajectories": len(expected),
        "assignment_count": len(assignments),
        "required_groups": list(GROUPS),
        "four_group_evaluable_count": evaluable,
        "threshold": MIN_EVALUABLE_REPAIRS,
        "errors": errors,
    }


def _leakage_check(
    diagnosis_run: Path,
    repair_run: Path,
    answer_key: Sequence[Mapping[str, Any]],
    shape: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify serialized generation requests exclude restricted labels and answers."""
    requests: list[Mapping[str, Any]] = []
    diagnosis_paths = sorted((Path(diagnosis_run) / "requests").glob("*.json"))
    for path in diagnosis_paths:
        value = _load_object(path)
        requests.append(value)
    repair_rows = _load_rows(Path(repair_run) / "restricted" / "requests.json")
    requests.extend(row["request"] for row in repair_rows if isinstance(row.get("request"), Mapping))
    violations: list[str] = []
    private_text = _private_text_values(answer_key)
    for index, request in enumerate(requests):
        keys = _nested_keys(request)
        forbidden = sorted(keys & _FORBIDDEN_REQUEST_KEYS)
        if forbidden:
            violations.append(f"request {index} contains restricted fields: {', '.join(forbidden)}")
        serialized = json.dumps(request, ensure_ascii=False, sort_keys=True)
        leaked = [value for value in private_text if value in serialized]
        if leaked:
            violations.append(f"request {index} contains restricted answer material")
    if len(requests) != 3 * shape["observed"]["trajectory_count"] + 4 * len(shape["error_sample_ids"]):
        violations.append("request artifact count does not match the frozen protocol")
    return {
        "status": "passed" if not violations else "failed",
        "request_count": len(requests),
        "status_detail": "zero restricted fields or answer values observed",
        "violations": violations,
    }


def _ledger_check(diagnosis_run: Path, repair_run: Path) -> dict[str, Any]:
    """Check durable call identity, response ownership, and bounded attempt rows."""
    details = {}
    errors: list[str] = []
    for name, run_dir in (("diagnosis", diagnosis_run), ("repair", repair_run)):
        expected_ids = _expected_call_ids(name, Path(run_dir))
        summary, run_errors = _ledger_summary(Path(run_dir) / "run.sqlite3", expected_ids)
        details[name] = summary
        errors.extend(f"{name}: {error}" for error in run_errors)
    return {"status": "passed" if not errors else "failed", "runs": details, "errors": errors}


def _restart_check(
    diagnosis_run: Path,
    repair_run: Path,
    restart_record: Mapping[str, Any] | Path | None,
) -> dict[str, Any]:
    """Reopen both ledgers and report persistence without claiming a live resume."""
    errors: list[str] = []
    snapshots = {}
    for name, run_dir in (("diagnosis", diagnosis_run), ("repair", repair_run)):
        path = Path(run_dir) / "run.sqlite3"
        first = _ledger_snapshot(path)
        second = _ledger_snapshot(path)
        if first != second:
            errors.append(f"{name}: ledger changed between reopen reads")
        snapshots[name] = {"analysis_results": first["analysis_results"]}
    provided = _load_object(restart_record) if restart_record is not None else None
    resume_execution = "not_run"
    if provided is not None:
        if provided.get("status") != "passed":
            errors.append("provided restart record is not passed")
        else:
            resume_execution = "provided_record"
    return {
        "status": "passed" if not errors else "failed",
        "persistence_reopen": "passed" if not errors else "failed",
        "resume_execution": resume_execution,
        "snapshots": snapshots,
        "errors": errors,
        "scope_note": "Reopen integrity is verified locally; no provider restart is claimed without a record.",
    }


def _offline_recompute(
    audit: Mapping[str, Any],
    diagnosis_run: Path,
    diagnosis_analysis: Mapping[str, Any],
    repair_run: Path,
    repair_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild both metric reports from raw artifacts and compare canonical JSON."""
    diagnosis_run_path = Path(diagnosis_run) / "run.json"
    diagnosis_recomputed = analyze_diagnoses(diagnosis_run_path, audit)
    diagnosis_match = canonical_json(diagnosis_recomputed) == canonical_json(diagnosis_analysis)
    assignments = _load_rows(Path(repair_run) / "restricted" / "assignments.json")
    evaluations = _load_jsonl(Path(repair_run) / "blind" / "evaluations.jsonl")
    bootstrap = repair_analysis.get("bootstrap", {})
    repair_recomputed = analyze_repairs(
        evaluations,
        assignments,
        bootstrap_samples=int(bootstrap.get("samples", 1000)),
        seed=int(bootstrap.get("seed", 0)),
    )
    repair_match = canonical_json(repair_recomputed) == canonical_json(repair_analysis)
    errors = []
    if not diagnosis_match:
        errors.append("diagnosis analysis does not match offline recomputation")
    if not repair_match:
        errors.append("repair analysis does not match offline recomputation")
    return {
        "status": "passed" if not errors else "failed",
        "diagnosis_match": diagnosis_match,
        "repair_match": repair_match,
        "errors": errors,
    }
