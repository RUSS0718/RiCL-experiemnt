"""Offline validation and analysis for the frozen MS3 confirmatory protocol."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..analysis.ms3_repair import analyze_repairs
from .ms3.analysis import analyze_diagnoses
from .ms3.protocol import METHOD_IDS
from .ms3_repair import GROUPS


def build_confirmatory_report(
    diagnosis_run: Mapping[str, Any] | Path,
    diagnosis_audit: Mapping[str, Any] | Path,
    repair_assignments: Iterable[Mapping[str, Any]],
    repair_evaluations: Iterable[Mapping[str, Any]],
    *,
    expected_run_id: str | None = None,
    expected_manifest_fingerprint: str | None = None,
    expected_config: Mapping[str, Any] | None = None,
    real_api_verified: bool | None = None,
    human_review_verified: bool | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Recompute H3a/H3b from restricted artifacts without calling a model."""
    run = _load_object(diagnosis_run)
    audit = _load_object(diagnosis_audit)
    assignments = [dict(row) for row in repair_assignments]
    evaluations = [dict(row) for row in repair_evaluations]
    _validate_identity(run, audit, expected_run_id, expected_manifest_fingerprint)
    normalized_audit = _normalize_audit(audit)
    h3a = analyze_diagnoses(run, normalized_audit)
    h3b = analyze_repairs(
        evaluations,
        assignments,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    coverage = _coverage(run, assignments, evaluations)
    identity = _identity_report(run, audit, assignments, expected_run_id, expected_manifest_fingerprint)
    configuration = _configuration_report(run, assignments, expected_config)
    external_evidence = _external_evidence(real_api_verified, human_review_verified)
    status = "ready_for_external_signoff" if not external_evidence["blocking_reasons"] else "blocked_external_evidence"
    return {
        "schema_version": 1,
        "study": "ms3",
        "phase": "confirmatory",
        "status": status,
        "run_id": run.get("run_id"),
        "manifest_id": run.get("manifest_id"),
        "manifest_fingerprint": run.get("manifest_fingerprint"),
        "identity": identity,
        "configuration": configuration,
        "coverage": coverage,
        "h3a": h3a,
        "h3b": h3b,
        "strata": h3b.get("position_offset", {}),
        "external_evidence": external_evidence,
        "interpretation_boundary": (
            "These results describe fixed-prompt diagnosis and position-prompt repair effects; "
            "they do not establish autonomous error localization or repair."
        ),
    }


def write_confirmatory_report(
    diagnosis_run: Path,
    diagnosis_audit: Path,
    repair_assignments: Path,
    repair_evaluations: Path,
    output: Path,
    **kwargs: Any,
) -> Path:
    """Write one deterministic JSON report from already persisted artifacts."""
    assignments = _load_jsonl(repair_assignments)
    evaluations = _load_jsonl(repair_evaluations)
    report = build_confirmatory_report(
        diagnosis_run, diagnosis_audit, assignments, evaluations, **kwargs
    )
    if output.exists():
        raise ValueError(f"confirmatory report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def _validate_identity(
    run: Mapping[str, Any],
    audit: Mapping[str, Any],
    expected_run_id: str | None,
    expected_manifest_fingerprint: str | None,
) -> None:
    """Reject drift between the frozen run, restricted audit, and caller expectations."""
    run_id = _required_text(run.get("run_id"), "run_id")
    run_fingerprint = _required_text(run.get("manifest_fingerprint"), "manifest fingerprint")
    audit_fingerprint = _required_text(audit.get("manifest_fingerprint"), "audit manifest fingerprint")
    if run_fingerprint != audit_fingerprint:
        raise ValueError("run and audit manifest fingerprint do not match")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("run_id does not match the frozen confirmatory run")
    if expected_manifest_fingerprint is not None and run_fingerprint != expected_manifest_fingerprint:
        raise ValueError("manifest fingerprint does not match the frozen confirmatory run")


def _normalize_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Accept the canonical label field and the explicit first-error alias."""
    result = dict(audit)
    labels = []
    for row in audit.get("processbench_labels", []):
        if not isinstance(row, Mapping):
            labels.append(row)
            continue
        value = dict(row)
        if "label" not in value and "first_error_step" in value:
            value["label"] = value["first_error_step"]
        labels.append(value)
    result["processbench_labels"] = labels
    return result


def _coverage(
    run: Mapping[str, Any],
    assignments: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count denominator, status, and missingness facts without hiding failures."""
    samples = run.get("samples") if isinstance(run.get("samples"), list) else []
    sample_ids = {str(row.get("sample_id")) for row in samples if isinstance(row, Mapping)}
    method_status_counts: dict[str, dict[str, int]] = {}
    methods = run.get("methods") if isinstance(run.get("methods"), Mapping) else {}
    for method in METHOD_IDS:
        method_status_counts[method] = dict(
            Counter(
                str(row.get("status"))
                for row in methods.get(method, [])
                if isinstance(row, Mapping)
            )
        )
    by_sample: dict[str, set[str]] = {}
    for row in assignments:
        by_sample.setdefault(str(row.get("sample_id")), set()).add(str(row.get("group")))
    expected_repair_samples = {sample for sample in by_sample if sample in sample_ids}
    complete = sum(groups == set(GROUPS) for groups in by_sample.values())
    missing_group_assignments = sum(len(set(GROUPS) - groups) for groups in by_sample.values())
    evaluation_ids = {str(row.get("evaluation_id")) for row in evaluations}
    missing_evaluations = sum(
        1 for row in assignments if str(row.get("evaluation_id")) not in evaluation_ids
    )
    return {
        "diagnosis": {
            "trajectory_count": len(samples),
            "method_status_counts": method_status_counts,
            "methods_with_complete_denominator": {
                method: sum(method_status_counts[method].values()) == len(samples)
                for method in METHOD_IDS
            },
        },
        "repair": {
            "assignment_sample_count": len(expected_repair_samples),
            "complete_four_group_samples": complete,
            "missing_group_assignments": missing_group_assignments,
            "missing_evaluations": missing_evaluations,
            "evaluation_status_counts": dict(
                Counter(str(row.get("status", "submitted")) for row in evaluations)
            ),
        },
    }


def _identity_report(
    run: Mapping[str, Any],
    audit: Mapping[str, Any],
    assignments: list[dict[str, Any]],
    expected_run_id: str | None,
    expected_manifest_fingerprint: str | None,
) -> dict[str, Any]:
    """Expose identity checks and any available assignment-level identity evidence."""
    assignment_run_ids = {str(row["run_id"]) for row in assignments if row.get("run_id") is not None}
    assignment_fingerprints = {
        str(row["manifest_fingerprint"])
        for row in assignments
        if row.get("manifest_fingerprint") is not None
    }
    return {
        "same_run_id": expected_run_id is None or run.get("run_id") == expected_run_id,
        "same_manifest_fingerprint": (
            expected_manifest_fingerprint is None
            or run.get("manifest_fingerprint") == expected_manifest_fingerprint
        ),
        "audit_matches_run": audit.get("manifest_fingerprint") == run.get("manifest_fingerprint"),
        "assignment_run_ids": sorted(assignment_run_ids),
        "assignment_manifest_fingerprints": sorted(assignment_fingerprints),
        "assignment_identity_complete": bool(assignment_run_ids and assignment_fingerprints),
    }


def _configuration_report(
    run: Mapping[str, Any],
    assignments: list[dict[str, Any]],
    expected_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare optional frozen settings and report absent evidence instead of guessing."""
    observed = run.get("effective_parameters")
    if observed is None:
        observed = run.get("parameters")
    assignment_parameters = [row.get("effective_parameters") for row in assignments if row.get("effective_parameters") is not None]
    checks = {
        "run_parameters_present": observed is not None,
        "assignment_parameters_present": bool(assignment_parameters),
    }
    if expected_config is not None:
        expected_parameters = expected_config.get("effective_parameters", expected_config.get("parameters"))
        checks["matches_expected_parameters"] = observed == expected_parameters
    else:
        checks["matches_expected_parameters"] = None
    return {"checks": checks, "observed_parameters": observed}


def _external_evidence(real_api_verified: bool | None, human_review_verified: bool | None) -> dict[str, Any]:
    """Represent external sign-off requirements explicitly and conservatively."""
    reasons = []
    if real_api_verified is not True:
        reasons.append("real API evidence is not verified")
    if human_review_verified is not True:
        reasons.append("human blind-review evidence is not verified")
    return {
        "real_api_verified": real_api_verified is True,
        "human_review_verified": human_review_verified is True,
        "blocking_reasons": reasons,
    }


def _load_object(value: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    loaded = json.loads(value.read_text(encoding="utf-8")) if isinstance(value, Path) else value
    if not isinstance(loaded, Mapping):
        raise TypeError("confirmatory artifacts must be JSON objects")
    return loaded


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {line_number} must be an object")
        rows.append(value)
    return rows


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value
