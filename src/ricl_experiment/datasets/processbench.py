"""ProcessBench trace validation and shared diagnosis/repair pool selection."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any


def validate_processbench_record(record: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Validate a trace's step-index convention and scoreability before selection."""
    reasons: list[str] = []
    steps = _steps(record.get("steps"))
    if len(steps) < 2 or any(not step for step in steps):
        reasons.append("insufficient_steps")

    label = record.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label < -1:
        reasons.append("invalid_label")
    elif label >= 0 and label >= len(steps):
        reasons.append("invalid_label_out_of_range")

    if not isinstance(record.get("problem"), str) or not record["problem"].strip():
        reasons.append("missing_problem")
    trajectory_id = record.get("trajectory_id")
    if trajectory_id is None:
        trajectory_id = record.get("id")
    if trajectory_id is None or not str(trajectory_id).strip():
        reasons.append("missing_trajectory_id")
    if not isinstance(record.get("final_answer_correct"), bool):
        reasons.append("invalid_final_answer_correct")

    if (
        isinstance(label, int)
        and not isinstance(label, bool)
        and label >= 0
        and not _has_independent_scoring_basis(record)
    ):
        reasons.append("missing_independent_scoring_basis")

    return not reasons, reasons


def build_processbench_pools(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build label-free study pools and a separate, restricted label audit."""
    diagnosis: list[dict[str, Any]] = []
    repair: list[dict[str, Any]] = []
    label_audit: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for record in records:
        trajectory_id = _trajectory_id(record)
        valid, reasons = validate_processbench_record(record)
        if not valid:
            exclusions.append({"trajectory_id": trajectory_id, "reasons": reasons})
            continue
        label = record["label"]
        meaning = "no_error" if label == -1 else "first_error_step_index_zero_based"
        row = {
            "trajectory_id": trajectory_id,
            "question_group_id": _question_group_id(record),
        }
        diagnosis.append(row)
        label_audit.append(
            {
                "trajectory_id": trajectory_id,
                "label": label,
                "meaning": meaning,
                "final_answer_correct": record["final_answer_correct"],
            }
        )
        if label >= 0:
            repair.append(dict(row))

    return {
        "diagnosis": diagnosis,
        "repair": repair,
        "label_audit": label_audit,
        "exclusions": exclusions,
    }


def _trajectory_id(record: Mapping[str, Any]) -> str:
    """Return the stable trajectory identifier, preferring an explicit trace ID."""
    value = record.get("trajectory_id")
    if value is None or not str(value).strip():
        value = record.get("id")
    return "" if value is None else str(value).strip()


def _question_group_id(record: Mapping[str, Any]) -> str:
    """Reuse an upstream group ID or the manifest's exact normalized-problem fallback."""
    for key in ("question_group_id", "original_question_group_id", "canonical_question_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    problem = record.get("problem")
    normalized = " ".join(problem.split()) if isinstance(problem, str) else ""
    return "problem-" + hashlib.sha256(normalized.encode()).hexdigest()


def _has_independent_scoring_basis(record: Mapping[str, Any]) -> bool:
    material = record.get("reference_material")
    if not isinstance(material, Mapping):
        return False
    return (
        material.get("kind") in {"answer", "solution"}
        and isinstance(material.get("content"), str)
        and bool(material["content"].strip())
        and isinstance(material.get("source"), str)
        and bool(material["source"].strip())
        and material.get("independently_verified") is True
    )


def _steps(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [step.strip() if isinstance(step, str) else "" for step in value]
    if isinstance(value, str):
        return [step.strip() for step in value.splitlines() if step.strip()]
    return []
