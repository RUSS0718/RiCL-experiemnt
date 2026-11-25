"""Offline MS1 sampling and blinded annotation artifact contracts."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .schemas import MS2Trace

MS1_PLAN_SCHEMA_VERSION = 1
MS1_ANNOTATION_SCHEMA_VERSION = 1
_SAMPLE_OUTCOMES = ("incorrect", "correct")
_ANNOTATION_FORBIDDEN_KEYS = {
    "answer_key",
    "expert_label",
    "final_answer_correct",
    "known_outcome",
    "reference_answer",
    "sampling_outcome",
}
_VALIDITY_STATUSES = {"Valid", "Invalid", "Unsupported", "Unknown"}
_DEPENDENCY_STATUSES = {"confirmed", "excluded", "unknown"}


def build_ms1_sampling_plan(
    traces: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    independent_outcomes: Iterable[Mapping[str, Any]] | None = None,
    failure_target: int = 20,
    correct_reference_target: int = 10,
) -> dict[str, Any]:
    """Build a deterministic, source/outcome-stratified path sampling plan.

    The outcome sidecar is required to be independent of the MS2 runner's answer
    heuristic. This prevents a trace's built-in answer key reference from being
    mistaken for the independent final judgment required by the MS1 protocol.
    """
    _require_nonnegative_seed(seed)
    _require_nonnegative_target(failure_target, "failure_target")
    _require_nonnegative_target(correct_reference_target, "correct_reference_target")
    if independent_outcomes is None:
        raise ValueError("independent_outcomes sidecar is required for MS1 sampling")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for trace_index, raw_trace in enumerate(traces):
        trace = MS2Trace.model_validate(raw_trace)
        for sample in trace.samples:
            _collect_sample_paths(trace, sample, trace_index, eligible, excluded)
    outcomes = _normalize_independent_outcomes(independent_outcomes)
    eligible = _join_independent_outcomes(eligible, outcomes, excluded)

    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    group_conflicts = 0
    targets = {"incorrect": failure_target, "correct": correct_reference_target}
    randomizer = random.Random(seed)
    for outcome in _SAMPLE_OUTCOMES:
        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            if row["sampling_outcome"] == outcome:
                by_stratum[row["stratum"]].append(row)
        for rows in by_stratum.values():
            rows.sort(key=lambda row: row["path_id"])
            randomizer.shuffle(rows)
        pools = {name: list(rows) for name, rows in sorted(by_stratum.items())}
        selected_for_outcome = 0
        while selected_for_outcome < targets[outcome]:
            progress = False
            for stratum in sorted(pools):
                pool = pools[stratum]
                while pool and pool[0]["question_group_id"] in selected_groups:
                    pool.pop(0)
                    group_conflicts += 1
                if not pool:
                    continue
                row = dict(pool.pop(0))
                selected_groups.add(row["question_group_id"])
                row["selection_order"] = len(selected)
                selected.append(row)
                selected_for_outcome += 1
                progress = True
                if selected_for_outcome == targets[outcome]:
                    break
            if not progress:
                break

    selected_by_path = {row["path_id"] for row in selected}
    return {
        "schema_version": MS1_PLAN_SCHEMA_VERSION,
        "plan_type": "ms1_dependency_sampling",
        "seed": seed,
        "targets": targets,
        "max_per_question_group": 1,
        "selected_paths": selected,
        "excluded": excluded,
        "strata": _strata_report(eligible, selected_by_path),
        "deficits": {
            outcome: targets[outcome] - sum(row["sampling_outcome"] == outcome for row in selected)
            for outcome in _SAMPLE_OUTCOMES
        },
        "denominators": {
            "candidate_paths": len(eligible),
            "selected_paths": len(selected),
            "question_groups": len({row["question_group_id"] for row in eligible}),
            "selected_question_groups": len(selected_groups),
            "excluded_records": len(excluded),
            "group_conflicts": group_conflicts,
        },
    }


def _collect_sample_paths(
    trace: MS2Trace,
    sample: Mapping[str, Any],
    trace_index: int,
    eligible: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> None:
    """Convert one real-shaped MS2 sample row into candidate rollout paths."""
    sample_id = _required_text(sample.get("sample_id"), "sample_id")
    question_group_id = sample.get("question_group_id")
    if not isinstance(question_group_id, str) or not question_group_id.strip():
        excluded.append({"sample_id": sample_id, "reason": "missing_question_group_id"})
        return
    candidates = sample.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        excluded.append({"sample_id": sample_id, "reason": "no_usable_rollout"})
        return
    source = str(sample.get("source") or sample.get("sampling_stratum") or "unknown")
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            excluded.append(
                {
                    "sample_id": sample_id,
                    "candidate_index": candidate_index,
                    "reason": "invalid_candidate",
                }
            )
            continue
        rollouts = candidate.get("rollouts")
        if not isinstance(rollouts, list):
            excluded.append(
                {
                    "sample_id": sample_id,
                    "candidate_index": candidate_index,
                    "reason": "missing_rollouts",
                }
            )
            continue
        candidate_id = str(candidate.get("candidate_id") or f"candidate-{candidate_index}")
        prefix = str(candidate.get("prefix_text") or "")
        for rollout_position, rollout in enumerate(rollouts):
            rollout_index = (
                rollout.get("rollout_index", rollout_position)
                if isinstance(rollout, Mapping)
                else rollout_position
            )
            if type(rollout_index) is not int or rollout_index < 0:
                rollout_index = rollout_position
            path = _path_identity(trace.run_id, sample_id, candidate_id, rollout_index, "eligible")
            if candidate.get("generation_status") == "failed":
                excluded.append(
                    {
                        **path,
                        "reason": "runtime_failure",
                        "failure": candidate.get("generation_failure"),
                    }
                )
                continue
            if not isinstance(rollout, Mapping):
                excluded.append({**path, "reason": "invalid_rollout"})
                continue
            text = rollout.get("text")
            rollout_status = rollout.get("status")
            if rollout_status != "completed":
                reason = {
                    "failed": "runtime_failure",
                    "not_dispatched": "rollout_not_dispatched",
                }.get(str(rollout_status), "no_usable_rollout")
                excluded.append({**path, "reason": reason})
                continue
            if not isinstance(text, str) or not text.strip():
                excluded.append({**path, "reason": "missing_rollout_text"})
                continue
            if candidate.get("prefix_status") != "available" or not prefix.strip():
                excluded.append({**path, "reason": "unsegmentable_path"})
                continue
            segments = _path_segments(candidate, rollout_index, prefix, text)
            if segments is None:
                excluded.append({**path, "reason": "unsegmentable_path"})
                continue
            eligible.append(
                {
                    **path,
                    "trace_index": trace_index,
                    "trace_run_id": trace.run_id,
                    "manifest_id": trace.manifest_id,
                    "phase": trace.phase,
                    "checkpoint_status": (
                        candidate.get("initial_evaluation", {}).get("status")
                        if isinstance(candidate.get("initial_evaluation"), Mapping)
                        else None
                    ),
                    "source": source,
                    "stratum": None,
                    "question_group_id": question_group_id,
                    "problem": str(sample.get("problem") or ""),
                    "checkpoint_text": prefix,
                    "rollout_text": text,
                    "candidate_index": candidate.get("candidate_index", candidate_index),
                    "checkpoint_id": candidate.get("checkpoint_id"),
                    "route_id": candidate.get("route_id"),
                    "source_record_id": sample.get("source_record_id"),
                    "revision": sample.get("revision"),
                    "split": sample.get("split"),
                    "segments": segments,
                    "termination": {
                        "stop_reason": candidate.get("stop_reason"),
                        "truncation_status": candidate.get("truncation_status"),
                        "last_step_status": candidate.get("last_step_status"),
                    },
                }
            )


def _normalize_independent_outcomes(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Require one explicit independent outcome for each supplied path."""
    result = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("independent outcome records must be objects")
        path_id = _required_text(record.get("path_id"), "independent outcome path_id")
        if path_id in result:
            raise ValueError(f"duplicate independent outcome for path {path_id}")
        status = record.get("status")
        if status not in {"correct", "incorrect", "unknown", "no_evaluable_output"}:
            raise ValueError(f"independent outcome for {path_id} has unsupported status")
        result[path_id] = {
            "status": status,
            "reference": record.get("reference"),
            "checkpoint_status": record.get("checkpoint_status"),
        }
    return result


def _join_independent_outcomes(
    paths: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    excluded: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join reviewed outcomes without consulting MS2's built-in answer heuristic."""
    eligible = []
    for path in paths:
        path_id = str(path["path_id"])
        if path.get("checkpoint_status") == "correct":
            excluded.append({**path, "reason": "checkpoint_completed_correct"})
            continue
        outcome = outcomes.get(path_id)
        if outcome is None:
            excluded.append({**path, "reason": "independent_outcome_missing"})
            continue
        if outcome.get("checkpoint_status") == "correct":
            excluded.append({**path, "reason": "checkpoint_completed_correct"})
            continue
        status = outcome.get("status")
        if status not in _SAMPLE_OUTCOMES:
            reason = (
                "independent_outcome_unknown"
                if status == "unknown"
                else "independent_outcome_no_evaluable_output"
            )
            excluded.append(
                {**path, "reason": reason, "outcome_reference": outcome.get("reference")}
            )
            continue
        if not outcome.get("reference"):
            excluded.append({**path, "reason": "independent_outcome_reference_missing"})
            continue
        eligible.append(
            {
                **path,
                "sampling_outcome": status,
                "stratum": f"{path['source']}|{status}",
                "outcome_reference": outcome.get("reference"),
            }
        )
    return eligible


def _path_identity(
    trace_run_id: str,
    sample_id: str,
    candidate_id: str,
    rollout_index: int,
    reason: str,
) -> dict[str, Any]:
    """Return a stable path identity for an excluded or selected rollout."""
    return {
        "path_id": f"{trace_run_id}:{sample_id}:{candidate_id}:rollout:{rollout_index}",
        "sample_id": sample_id,
        "candidate_id": candidate_id,
        "rollout_index": rollout_index,
        "reason": reason,
    }


def _path_segments(
    candidate: Mapping[str, Any], rollout_index: int, prefix: str, rollout_text: str
) -> list[dict[str, Any]] | None:
    """Return only the checkpoint and selected rollout when source bounds are reliable."""
    raw_segments = candidate.get("segments")
    if not isinstance(raw_segments, list):
        return None
    segments = [dict(segment) for segment in raw_segments if isinstance(segment, Mapping)]
    prefix_segments = [
        segment for segment in segments if segment.get("kind") == "checkpoint_prefix"
    ]
    rollout_segments = [
        segment for segment in segments if segment.get("kind") == f"rollout_{rollout_index}"
    ]
    if len(prefix_segments) != 1 or len(rollout_segments) != 1:
        return None
    prefix_segment, rollout_segment = prefix_segments[0], rollout_segments[0]
    if (
        prefix_segment.get("status") != "completed"
        or prefix_segment.get("start") != 0
        or prefix_segment.get("end") != len(prefix)
        or prefix_segment.get("text") != prefix
        or rollout_segment.get("status") != "completed"
        or rollout_segment.get("start") != len(prefix)
        or rollout_segment.get("end") != len(prefix) + len(rollout_text)
        or rollout_segment.get("text") != rollout_text
    ):
        return None
    return [prefix_segment, rollout_segment]


def build_blind_annotation_manifest(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Create an annotation input that contains no independent outcome labels."""
    selected = plan.get("selected_paths")
    if not isinstance(selected, list):
        raise TypeError("sampling plan is missing selected_paths")
    paths = []
    for path in selected:
        if not isinstance(path, Mapping):
            raise TypeError("selected path must be an object")
        paths.append(
            {
                "path_id": _required_text(path.get("path_id"), "path_id"),
                "sample_id": _required_text(path.get("sample_id"), "sample_id"),
                "candidate_id": _required_text(path.get("candidate_id"), "candidate_id"),
                "rollout_index": path.get("rollout_index"),
                "checkpoint_id": path.get("checkpoint_id"),
                "route_id": path.get("route_id"),
                "question_group_id": _required_text(
                    path.get("question_group_id"), "question_group_id"
                ),
                "source": str(path.get("source") or "unknown"),
                "problem": str(path.get("problem") or ""),
                "visible_text": f"{path.get('checkpoint_text', '')}{path.get('rollout_text', '')}",
                "segments": [
                    {
                        key: segment[key]
                        for key in ("kind", "status", "start", "end", "text")
                        if key in segment
                    }
                    for segment in path.get("segments", [])
                    if isinstance(segment, Mapping)
                ],
                "annotation_fields": [
                    "step_id",
                    "span",
                    "claim",
                    "applicability",
                    "dependency_ids",
                    "dependency_status",
                    "validity_status",
                    "verification_evidence",
                    "token_count",
                    "equivalence_group_id",
                    "is_nontrivial",
                ],
            }
        )
    _assert_blinded(paths)
    return {
        "schema_version": MS1_ANNOTATION_SCHEMA_VERSION,
        "manifest_type": "ms1_blind_annotation",
        "seed": plan.get("seed"),
        "paths": paths,
    }


def import_ms1_annotations(
    plan: Mapping[str, Any],
    independent_records: Iterable[Mapping[str, Any]],
    third_reviews: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate two independent annotation records and reconcile disagreements."""
    expected = {
        str(path["path_id"])
        for path in plan.get("selected_paths", [])
        if isinstance(path, Mapping) and path.get("path_id")
    }
    independent = _normalize_review_records(independent_records, expected, "independent")
    third = _normalize_review_records(third_reviews, expected, "third_reviewer")
    plan_by_path = {
        str(path["path_id"]): path
        for path in plan.get("selected_paths", [])
        if isinstance(path, Mapping) and path.get("path_id")
    }
    output = []
    for path_id in sorted(expected):
        reviews = independent.get(path_id, [])
        if len(reviews) != 2:
            raise ValueError(f"path {path_id} requires exactly two independent annotation records")
        if len({str(review["reviewer_id"]) for review in reviews}) != 2:
            raise ValueError(f"path {path_id} requires two distinct independent reviewers")
        third_review = third.get(path_id, [])
        if len(third_review) > 1:
            raise ValueError(f"path {path_id} has more than one third-reviewer record")
        consensus, conflicts, status = _reconcile_reviews(
            reviews[0], reviews[1], third_review[0] if third_review else None
        )
        plan_path = plan_by_path[path_id]
        context = {
            key: plan_path[key]
            for key in (
                "trace_index",
                "trace_run_id",
                "manifest_id",
                "phase",
                "sample_id",
                "candidate_id",
                "candidate_index",
                "checkpoint_id",
                "route_id",
                "rollout_index",
                "question_group_id",
                "source",
                "source_record_id",
                "revision",
                "split",
                "problem",
                "checkpoint_text",
                "rollout_text",
                "checkpoint_status",
                "segments",
                "termination",
            )
            if key in plan_path
        }
        output.append(
            {
                "path_id": path_id,
                "trace_context": context,
                "independent_reviews": reviews,
                "third_review": third_review[0] if third_review else None,
                "conflicts": conflicts,
                "reconciliation_status": status,
                "consensus": consensus,
            }
        )
    return {
        "schema_version": MS1_ANNOTATION_SCHEMA_VERSION,
        "artifact_type": "ms1_dependency_annotations",
        "seed": plan.get("seed"),
        "paths": output,
    }


def _normalize_review_records(
    records: Iterable[Mapping[str, Any]], expected: set[str], role: str
) -> dict[str, list[dict[str, Any]]]:
    """Normalize review records while retaining Unknown and Unsupported labels."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("annotation records must be objects")
        _assert_no_restricted_keys(raw)
        path_id = _required_text(raw.get("path_id"), "annotation path_id")
        if path_id not in expected:
            raise ValueError(f"annotation references an unselected path: {path_id}")
        reviewer_id = _required_text(raw.get("reviewer_id", raw.get("annotator_id")), "reviewer_id")
        nodes = raw.get("nodes")
        if not isinstance(nodes, list):
            raise TypeError(f"annotation {path_id} nodes must be an array")
        first_error = raw.get("first_error")
        first_error_step_id = raw.get("first_error_step_id")
        first_error_status = raw.get("first_error_status")
        if isinstance(first_error, Mapping):
            first_error_step_id = first_error.get("step_id", first_error_step_id)
            first_error_status = first_error.get("status", first_error_status)
        if first_error_status is None:
            first_error_status = "confirmed" if first_error_step_id else "unknown"
        if first_error_status not in {"confirmed", "none", "unknown"}:
            raise ValueError("first_error_status must be confirmed, none, or unknown")
        if first_error_step_id is not None:
            first_error_step_id = _required_text(first_error_step_id, "first_error_step_id")
        grouped[path_id].append(
            {
                "reviewer_id": reviewer_id,
                "role": role,
                "nodes": [_normalize_node(node) for node in nodes],
                "first_error_step_id": first_error_step_id,
                "first_error_status": first_error_status,
            }
        )
    return grouped


def _normalize_node(node: Any) -> dict[str, Any]:
    """Validate one dependency-graph node from a blind review export."""
    if not isinstance(node, Mapping):
        raise TypeError("annotation nodes must be objects")
    step_id = _required_text(node.get("step_id"), "step_id")
    span = node.get("span")
    start = node.get("start", span.get("start") if isinstance(span, Mapping) else None)
    end = node.get("end", span.get("end") if isinstance(span, Mapping) else None)
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ValueError(f"node {step_id} requires a valid half-open span")
    claim = _required_text(node.get("claim"), f"claim for {step_id}")
    validity = node.get("validity_status")
    if validity not in _VALIDITY_STATUSES:
        raise ValueError(f"node {step_id} has unsupported validity_status")
    dependencies = node.get("dependency_ids", [])
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) or not item.strip() for item in dependencies
    ):
        raise ValueError(f"node {step_id} dependency_ids must be string IDs")
    dependency_status = node.get("dependency_status", {})
    if not isinstance(dependency_status, Mapping) or any(
        key not in dependencies or value not in _DEPENDENCY_STATUSES
        for key, value in dependency_status.items()
    ):
        raise ValueError(f"node {step_id} dependency_status is invalid")
    token_count = node.get("token_count")
    if token_count is not None and (type(token_count) is not int or token_count < 0):
        raise ValueError(f"node {step_id} token_count must be a non-negative integer or null")
    is_nontrivial = node.get("is_nontrivial", True)
    if type(is_nontrivial) is not bool:
        raise ValueError(f"node {step_id} is_nontrivial must be a boolean")
    return {
        "step_id": step_id,
        "start": start,
        "end": end,
        "claim": claim,
        "applicability": str(node.get("applicability") or ""),
        "dependency_ids": list(dependencies),
        "dependency_status": dict(dependency_status),
        "validity_status": validity,
        "verification_evidence": str(node.get("verification_evidence") or ""),
        "token_count": token_count,
        "equivalence_group_id": node.get("equivalence_group_id"),
        "is_nontrivial": node.get("is_nontrivial", True),
    }


def _reconcile_reviews(
    left: Mapping[str, Any], right: Mapping[str, Any], third: Mapping[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Use the third reviewer for disagreements, otherwise preserve unresolved Unknowns."""
    conflicts = _review_conflicts(left, right)
    if third is not None:
        return dict(third), conflicts, "third_reviewer_reconciled"
    if not conflicts:
        return dict(left), [], "independent_agreement"
    consensus = dict(left)
    consensus["first_error_step_id"] = None
    consensus["first_error_status"] = "unknown"
    left_nodes = {node["step_id"]: node for node in left["nodes"]}
    right_nodes = {node["step_id"]: node for node in right["nodes"]}
    merged_nodes = []
    for step_id in sorted(set(left_nodes) | set(right_nodes)):
        node = dict(left_nodes.get(step_id) or right_nodes[step_id])
        if left_nodes.get(step_id) != right_nodes.get(step_id):
            node["validity_status"] = "Unknown"
            node["verification_evidence"] = "unresolved independent-review disagreement"
        merged_nodes.append(node)
    consensus["nodes"] = merged_nodes
    return consensus, conflicts, "pending_third_review"


def _review_conflicts(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Identify disagreements that can change the dependency metrics."""
    conflicts = []
    if left.get("first_error_step_id") != right.get("first_error_step_id") or left.get(
        "first_error_status"
    ) != right.get("first_error_status"):
        conflicts.append({"field": "first_error"})
    left_nodes = {node["step_id"]: node for node in left["nodes"]}
    right_nodes = {node["step_id"]: node for node in right["nodes"]}
    for step_id in sorted(set(left_nodes) | set(right_nodes)):
        lnode, rnode = left_nodes.get(step_id), right_nodes.get(step_id)
        if lnode is None or rnode is None:
            conflicts.append({"field": "nodes", "step_id": step_id})
            continue
        for field in (
            "start",
            "end",
            "claim",
            "applicability",
            "dependency_ids",
            "dependency_status",
            "validity_status",
            "equivalence_group_id",
            "is_nontrivial",
        ):
            if lnode.get(field) != rnode.get(field):
                conflicts.append({"field": field, "step_id": step_id})
    return conflicts


def _assert_blinded(paths: Sequence[Mapping[str, Any]]) -> None:
    """Reject restricted labels from the artifact sent to independent annotators."""
    _assert_no_restricted_keys(paths)


def _assert_no_restricted_keys(value: Any) -> None:
    """Reject known restricted fields recursively at a blind-review boundary."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _ANNOTATION_FORBIDDEN_KEYS or key_text.endswith(
                ("evaluation", "evaluation_reference", "answer_key_ref")
            ):
                raise ValueError(f"blind annotation artifact contains restricted field: {key}")
            _assert_no_restricted_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_restricted_keys(child)


def _strata_report(rows: Sequence[Mapping[str, Any]], selected: set[str]) -> dict[str, Any]:
    """Summarize candidate and selected counts without hiding sampling gaps."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stratum"])].append(row)
    return {
        name: {
            "candidate_count": len(items),
            "question_group_count": len({str(item["question_group_id"]) for item in items}),
            "selected_count": sum(str(item["path_id"]) in selected for item in items),
            "selected_question_group_count": len(
                {
                    str(item["question_group_id"])
                    for item in items
                    if str(item["path_id"]) in selected
                }
            ),
        }
        for name, items in sorted(grouped.items())
    }


def _required_text(value: Any, name: str) -> str:
    """Require a non-empty string at an artifact boundary."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_seed(seed: int) -> None:
    """Reject booleans and negative values that are not valid fixed seeds."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _require_nonnegative_target(value: int, name: str) -> None:
    """Validate a requested sample quota."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
