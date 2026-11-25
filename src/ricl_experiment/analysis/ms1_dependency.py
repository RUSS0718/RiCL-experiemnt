"""Offline MS1 dependency and re-derivation metrics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def union_span_length(spans: Iterable[tuple[int, int] | list[int]]) -> int:
    """Count the union of validated half-open integer spans."""
    normalized = sorted((_span(span) for span in spans), key=lambda item: (item[0], item[1]))
    total = 0
    current_start = current_end = None
    for start, end in normalized:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total if current_start is None else total + current_end - current_start


def analyze_ms1_dependency(
    annotations: Iterable[Mapping[str, Any]],
    rederivations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compute per-path dependency metrics and paired re-derivation counts."""
    annotation_rows = [dict(row) for row in annotations]
    rederivation_by_path: dict[str, dict[str, Any]] = {}
    for row in rederivations:
        path_id = str(row["path_id"])
        if path_id in rederivation_by_path:
            raise ValueError(f"duplicate re-derivation record for path {path_id}")
        rederivation_by_path[path_id] = dict(row)
    per_path = []
    for annotation in annotation_rows:
        consensus = annotation.get("consensus") or annotation
        metrics = _graph_metrics(consensus)
        paired = _rederivation_metrics(rederivation_by_path.get(str(annotation["path_id"])))
        per_path.append({"path_id": annotation["path_id"], **metrics, "rederivation": paired})
    return {
        "schema_version": 1,
        "analysis_type": "ms1_dependency_and_rederivation",
        "paths": per_path,
        "summary": _summary(per_path),
        "denominators": {
            "annotation_paths": len(per_path),
            "rederivation_records": len(rederivation_by_path),
            "evaluable_dependency_paths": sum(
                row["error_propagation_ratio"]["value"] is not None for row in per_path
            ),
            "evaluable_reusable_claim_paths": sum(
                row["reusable_claim_ratio"]["value"] is not None for row in per_path
            ),
            "evaluable_rederivation_paths": sum(
                row["rederivation"]["status"] == "completed" for row in per_path
            ),
        },
    }


def _graph_metrics(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate propagation, depth, and reusable-claim metrics for one consensus."""
    nodes = [dict(node) for node in annotation.get("nodes", [])]
    node_by_id = {str(node["step_id"]): node for node in nodes}
    order = {str(node["step_id"]): index for index, node in enumerate(nodes)}
    first_error = annotation.get("first_error_step_id")
    first_error_status = annotation.get("first_error_status", "unknown")
    edges, unresolved_edges = _edges(nodes, node_by_id)
    first_error_id = str(first_error) if first_error else None
    first_error_present = first_error_id in order if first_error_id else False
    depths = (
        _depths(first_error_id, edges)
        if first_error_status == "confirmed" and first_error_present
        else {}
    )
    later_nodes = {
        step_id
        for step_id, index in order.items()
        if first_error_present and index > order[first_error_id]
    }
    affected = sorted((step_id for step_id in depths if step_id in later_nodes), key=order.get)
    denominator = len(later_nodes)
    if first_error_status != "confirmed":
        not_applicable_reason = "first_error_unresolved"
    elif not first_error_present:
        not_applicable_reason = "first_error_anchor_missing"
    elif not later_nodes:
        not_applicable_reason = "no_following_nodes"
    else:
        not_applicable_reason = None
    propagation = {
        "numerator": len(affected),
        "denominator": denominator,
        "value": len(affected) / denominator if denominator else None,
        "unresolved_dependency_edges": unresolved_edges,
        "not_applicable_reason": not_applicable_reason,
    }
    depth_reason = None if affected else (not_applicable_reason or "no_confirmed_affected_nodes")
    equivalent = _equivalent_claims(nodes, node_by_id, edges)
    reusable = sum(item["reusable"] for item in equivalent.values())
    claim_ratio = {
        "numerator": reusable,
        "denominator": len(equivalent),
        "value": reusable / len(equivalent) if equivalent else None,
        "not_applicable_reason": None if equivalent else "no_nontrivial_claims",
    }
    return {
        "error_propagation_ratio": propagation,
        "dependency_depth": {
            "by_step": {step_id: depths[step_id] for step_id in affected},
            "observed_nodes": len(affected),
            "unresolved_dependency_edges": unresolved_edges,
            "not_applicable_reason": depth_reason,
        },
        "reusable_claim_ratio": claim_ratio,
        "claim_groups": equivalent,
    }


def _edges(
    nodes: list[dict[str, Any]], node_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, set[str]], int]:
    """Build confirmed dependency edges and count unresolved dependency references."""
    edges: dict[str, set[str]] = {str(node["step_id"]): set() for node in nodes}
    unresolved = 0
    for node in nodes:
        child = str(node["step_id"])
        dependencies = node.get("dependency_ids", [])
        statuses = node.get("dependency_status", {})
        for dependency in dependencies:
            parent = str(dependency)
            status = (
                statuses.get(parent, "confirmed") if isinstance(statuses, Mapping) else "confirmed"
            )
            if parent not in node_by_id or status == "unknown":
                unresolved += 1
            elif status == "confirmed":
                edges.setdefault(parent, set()).add(child)
    return edges, unresolved


def _depths(first_error: str, edges: Mapping[str, set[str]]) -> dict[str, int]:
    """Return shortest confirmed edge distances from the first error."""
    distances = {first_error: 0}
    frontier = [first_error]
    while frontier:
        parent = frontier.pop(0)
        for child in sorted(edges.get(parent, ())):
            if child not in distances:
                distances[child] = distances[parent] + 1
                frontier.append(child)
    return distances


def _equivalent_claims(
    nodes: list[dict[str, Any]],
    node_by_id: Mapping[str, Mapping[str, Any]],
    edges: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    """Deduplicate claims while retaining whether a clean valid derivation exists."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        claim = str(node.get("claim") or "").strip()
        if not claim or node.get("is_nontrivial", True) is False:
            continue
        group = str(node.get("equivalence_group_id") or node["step_id"])
        groups.setdefault(group, []).append(node)
    invalid_ids = {
        str(node["step_id"]) for node in nodes if node.get("validity_status") == "Invalid"
    }
    result = {}
    for group, group_nodes in sorted(groups.items()):
        reusable = any(
            node.get("validity_status") == "Valid"
            and bool(str(node.get("verification_evidence") or "").strip())
            and not _depends_on_any(str(node["step_id"]), invalid_ids, node_by_id)
            and not _has_unknown_dependency(str(node["step_id"]), node_by_id)
            for node in group_nodes
        )
        result[group] = {
            "reusable": int(reusable),
            "occurrences": [str(node["step_id"]) for node in group_nodes],
        }
    return result


def _depends_on_any(
    step_id: str, targets: set[str], nodes: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Follow confirmed dependencies to detect contamination by an invalid node."""
    seen: set[str] = set()
    node = nodes.get(step_id)
    if node is None:
        return False
    statuses = node.get("dependency_status", {})
    frontier = [
        dependency
        for dependency in node.get("dependency_ids", [])
        if not isinstance(statuses, Mapping)
        or statuses.get(str(dependency), "confirmed") == "confirmed"
    ]
    while frontier:
        parent = str(frontier.pop())
        if parent in seen:
            continue
        seen.add(parent)
        if parent in targets:
            return True
        parent_node = nodes.get(parent)
        if parent_node is None:
            continue
        parent_statuses = parent_node.get("dependency_status", {})
        frontier.extend(
            dependency
            for dependency in parent_node.get("dependency_ids", [])
            if not isinstance(parent_statuses, Mapping)
            or parent_statuses.get(str(dependency), "confirmed") == "confirmed"
        )
    return False


def _has_unknown_dependency(step_id: str, nodes: Mapping[str, Mapping[str, Any]]) -> bool:
    """Treat unresolved support anywhere in a confirmed dependency chain as non-reusable."""
    seen: set[str] = set()
    frontier = [step_id]
    while frontier:
        current_id = str(frontier.pop())
        if current_id in seen:
            continue
        seen.add(current_id)
        current = nodes.get(current_id)
        if current is None:
            return True
        dependencies = current.get("dependency_ids", [])
        statuses = current.get("dependency_status", {})
        for dependency in dependencies:
            dependency_id = str(dependency)
            status = (
                statuses.get(dependency_id, "confirmed")
                if isinstance(statuses, Mapping)
                else "confirmed"
            )
            if status == "unknown":
                return True
            if status == "confirmed":
                if dependency_id not in nodes:
                    return True
                frontier.append(dependency_id)
    return False


def _rederivation_metrics(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Measure visible repeated spans separately from hidden reasoning usage."""
    if record is None:
        return {"status": "missing", "reason": "rederivation_record_absent"}
    if record.get("status") != "completed":
        return {"status": str(record.get("status") or "missing"), "reason": record.get("reason")}
    token_count_estimated = False
    if isinstance(record.get("token_spans"), list):
        repeated = union_span_length(record["token_spans"])
    else:
        text = str(record.get("visible_text") or "")
        character_spans = record.get("character_spans", [])
        repeated = _estimated_span_token_count(text, character_spans)
        token_count_estimated = True
    visible_total = record.get("visible_output_tokens")
    if type(visible_total) is not int or visible_total < 0:
        visible_total = _estimate_token_count(str(record.get("visible_text") or ""))
        token_count_estimated = True
    hidden = record.get("hidden_usage_tokens")
    hidden_status = "known" if type(hidden) is int and hidden >= 0 else "unknown"
    if hidden_status == "unknown" and isinstance(record.get("hidden_usage"), Mapping):
        hidden_record = record["hidden_usage"]
        candidate = hidden_record.get("tokens")
        if type(candidate) is int and candidate >= 0:
            hidden, hidden_status = candidate, "known"
    if hidden_status == "unknown" and isinstance(record.get("usage"), Mapping):
        candidate = record["usage"].get("reasoning_tokens")
        if type(candidate) is int and candidate >= 0:
            hidden, hidden_status = candidate, "known"
    return {
        "status": "completed",
        "rederived_claim_tokens": repeated,
        "visible_output_tokens": visible_total,
        "rederived_claim_ratio": repeated / visible_total if visible_total else None,
        "token_count_estimated": token_count_estimated,
        "hidden_usage": {
            "status": hidden_status,
            "tokens": hidden if hidden_status == "known" else None,
        },
        "matched_span_count": len(record.get("token_spans", record.get("character_spans", []))),
    }


def _estimated_span_token_count(text: str, spans: Any) -> int:
    """Count estimated tokens overlapping character spans."""
    boundaries = _token_boundaries(text)
    indices = {
        index
        for span in spans
        if isinstance(span, (list, tuple)) and len(span) == 2
        for index, (start, end) in enumerate(boundaries)
        if _span(span)[0] < end and _span(span)[1] > start
    }
    return len(indices)


def _estimate_token_count(text: str) -> int:
    """Use a documented regex tokenizer when a model tokenizer is unavailable."""
    return len(_token_boundaries(text))


def _token_boundaries(text: str) -> list[tuple[int, int]]:
    """Return deterministic approximate token boundaries for controlled offline fixtures."""
    import re

    return [(match.start(), match.end()) for match in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)]


def _span(value: Any) -> tuple[int, int]:
    """Validate one non-empty half-open integer span."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("spans must contain two integer offsets")
    start, end = value
    if start < 0 or end < start:
        raise ValueError("spans must be non-negative half-open intervals")
    return start, end


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate numerators, denominators, and dependency depths."""
    propagation_num = sum(row["error_propagation_ratio"]["numerator"] for row in rows)
    propagation_den = sum(row["error_propagation_ratio"]["denominator"] for row in rows)
    reusable_num = sum(row["reusable_claim_ratio"]["numerator"] for row in rows)
    reusable_den = sum(row["reusable_claim_ratio"]["denominator"] for row in rows)
    depths = [depth for row in rows for depth in row["dependency_depth"]["by_step"].values()]
    rederivation = [
        row["rederivation"] for row in rows if row["rederivation"]["status"] == "completed"
    ]
    repeated = sum(row["rederived_claim_tokens"] for row in rederivation)
    visible = sum(row["visible_output_tokens"] for row in rederivation)
    return {
        "error_propagation_ratio": _ratio(propagation_num, propagation_den),
        "reusable_claim_ratio": _ratio(reusable_num, reusable_den),
        "dependency_depth": {
            "observed_nodes": len(depths),
            "distribution": {str(value): depths.count(value) for value in sorted(set(depths))},
        },
        "rederived_claim_tokens": {
            "rederived_claim_tokens": repeated,
            "visible_output_tokens": visible,
            "ratio": repeated / visible if visible else None,
            "estimated_paths": sum(row["token_count_estimated"] for row in rederivation),
            "hidden_usage_tokens": sum(
                row["hidden_usage"]["tokens"] or 0
                for row in rederivation
                if row["hidden_usage"]["status"] == "known"
            ),
        },
    }


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    """Keep metric numerators and denominators visible in every report."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def write_ms1_case_export(report: Mapping[str, Any], path_ids: Iterable[str], output: Path) -> Path:
    """Write exactly two or three selected metric cases as a stable JSON artifact."""
    requested = [str(path_id) for path_id in path_ids]
    if len(requested) not in {2, 3}:
        raise ValueError("case export requires 2 or 3 path IDs")
    if len(set(requested)) != len(requested):
        raise ValueError("case export path IDs must be unique")
    by_path = {
        str(row["path_id"]): row
        for row in report.get("paths", [])
        if isinstance(row, Mapping) and row.get("path_id")
    }
    missing = [path_id for path_id in requested if path_id not in by_path]
    if missing:
        raise ValueError("case export path is absent from analysis: " + ", ".join(missing))
    payload = {
        "schema_version": 1,
        "artifact_type": "ms1_case_export",
        "case_count": len(requested),
        "cases": [by_path[path_id] for path_id in requested],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output
