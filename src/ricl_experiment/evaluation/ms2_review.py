"""Export blind MS2 adjudications and import them into a derived offline trace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import fingerprint
from ..evaluation.answer import AnswerDecision, AnswerStatus, apply_review_outcome


def _read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _write_new(path: Path, value: dict) -> None:
    output = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(output)


def _index(rows: list[dict], field: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row.get(field)
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"missing or duplicate {field}: {key}")
        result[key] = row
    return result


def export_answer_review(run_dir: Path, output: Path) -> None:
    """Build a blind package whose ID map is bound to the immutable run trace."""
    source = run_dir / "traces" / "ms2.json"
    data = source.read_bytes()
    trace = json.loads(data)
    references = _index(_read(run_dir / "restricted" / "answer-key.json"), "sample_id")
    tasks, bindings = [], {}
    for sample in trace["samples"]:
        sample_id = sample["sample_id"]
        reference = references[sample_id]
        context = {
            "problem": sample["problem"],
            "reference_answer": reference["reference_answer"],
            "reference_solution": reference.get("reference_solution"),
        }
        for candidate in sample["candidates"]:
            route = {"sample_id": sample_id, "route_id": candidate["route_id"]}
            if candidate.get("checkpoint_review_required"):
                rid = fingerprint(
                    [trace["run_id"], candidate["checkpoint_id"], "checkpoint-review"]
                )
                tasks.append(
                    {
                        "review_id": rid,
                        "kind": "checkpoint",
                        **context,
                        "checkpoint": candidate["prefix_text"],
                    }
                )
                bindings[rid] = {
                    **route,
                    "kind": "checkpoint",
                    "prefix_fingerprint": candidate["prefix_fingerprint"],
                }
            for rollout in candidate["rollouts"]:
                if (rollout.get("evaluation") or {}).get("status") != "blind_review":
                    continue
                rid = fingerprint([trace["run_id"], rollout["logical_call_id"], "answer-review"])
                tasks.append(
                    {
                        "review_id": rid,
                        "kind": "continuation",
                        **context,
                        "candidate_output": rollout["text"],
                    }
                )
                bindings[rid] = {
                    **route,
                    "kind": "continuation",
                    "logical_call_id": rollout["logical_call_id"],
                    "response_ref": rollout["response_ref"],
                }
    # Hash order hides source/candidate ordering without introducing another seed.
    tasks.sort(key=lambda row: row["review_id"])
    output.mkdir(parents=True, exist_ok=False)
    _write_new(
        output / "restricted/answer-review-audit.json",
        {
            "run_id": trace["run_id"],
            "source_trace_sha256": hashlib.sha256(data).hexdigest(),
            "tasks": bindings,
        },
    )
    _write_new(
        output / "blind/answer-review.json",
        {
            "instructions": (
                "For a Checkpoint classify a stated final answer, or use no_final_answer "
                "if none is stated. For a continuation judge its final answer. "
                "Record unknown where the verdict cannot be resolved."
            ),
            "tasks": tasks,
        },
    )


def import_answer_reviews(
    run_dir: Path, reviews: Path, output: Path, *, bundle: Path | None = None
) -> None:
    """Join signed Codex or human verdicts to immutable response IDs in a fresh trace."""
    source = run_dir / "traces/ms2.json"
    source_bytes = source.read_bytes()
    trace = json.loads(source_bytes)
    bundle = bundle or run_dir
    tasks = _index(_read(bundle / "blind/answer-review.json")["tasks"], "review_id")
    audit = _read(bundle / "restricted/answer-review-audit.json")
    bindings = audit.get("tasks", audit.get("records"))
    if not isinstance(bindings, dict) or tasks.keys() != bindings.keys():
        raise ValueError("blind tasks and audit identities disagree")
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    if audit.get("run_id", trace["run_id"]) != trace["run_id"] or (
        audit.get("source_trace_sha256", source_digest) != source_digest
    ):
        raise ValueError("blind packet does not belong to this persisted trace")
    by_sample = _index(trace["samples"], "sample_id")
    by_reference = _index(_read(run_dir / "restricted/answer-key.json"), "sample_id")
    responses = reviews.read_bytes()
    submitted = json.loads(responses)
    kind = submitted.get("reviewer_kind")
    if kind not in {"human", "codex"}:
        raise ValueError("reviewer_kind must be human or codex")
    verdicts = _index(submitted["reviews"], "review_id")
    for rid, verdict in verdicts.items():
        if rid not in tasks:
            raise ValueError(f"review ID is not present in the blind package: {rid}")
        if any(
            not isinstance(verdict.get(key), str) or not verdict[key].strip()
            for key in ("reviewer_id", "rationale")
        ):
            raise ValueError(f"reviewer_id and rationale are required: {rid}")
        task, binding = tasks[rid], bindings[rid]
        sample_id = binding["sample_id"]
        sample = by_sample.get(sample_id)
        reference = by_reference.get(sample_id)
        if (
            sample is None
            or reference is None
            or task["problem"] != sample["problem"]
            or (
                task["reference_answer"] != reference["reference_answer"]
                or task.get("reference_solution") != reference.get("reference_solution")
            )
        ):
            raise ValueError(f"blind task does not match the frozen problem and reference: {rid}")
        field = "route_id" if "route_id" in binding else "candidate_id"
        candidate = _index(sample["candidates"], field).get(binding[field])
        if candidate is None:
            raise ValueError(f"review does not bind to a frozen candidate: {rid}")
        record = {
            "review_id": rid,
            "reviewer_kind": kind,
            "reviewer_id": verdict["reviewer_id"],
            "review_rationale": verdict["rationale"],
        }
        if binding.get("kind") == "checkpoint":
            _apply_checkpoint(candidate, task, binding, verdict["verdict"], record)
        else:
            _apply_continuation(run_dir, candidate, task, binding, verdict["verdict"], record)
    if any(
        candidate.get("checkpoint_review_required") and "checkpoint_review" not in candidate
        for sample in trace["samples"]
        for candidate in sample["candidates"]
    ):
        raise ValueError("every available MS2 v2 Checkpoint needs an adjudication before import")
    pending_ids = sorted(tasks.keys() - verdicts.keys())
    trace["offline_review"] = {
        "status": "partial" if pending_ids else "complete",
        "expected_count": len(tasks),
        "pending_review_ids": pending_ids,
        "source_trace_sha256": source_digest,
        "review_file_sha256": hashlib.sha256(responses).hexdigest(),
        "reviewer_kind": kind,
        "imported_count": len(verdicts),
    }
    _write_new(output, trace)


def read_reviewed_trace(frozen_bytes: bytes, reviewed_path: Path) -> dict:
    """Check the provenance of an offline import before using it in further analysis."""
    trace = _read(reviewed_path)
    source = json.loads(frozen_bytes)
    digest = hashlib.sha256(frozen_bytes).hexdigest()
    if (
        trace.get("offline_review", {}).get("source_trace_sha256") != digest
        or trace.get("run_id") != source["run_id"]
    ):
        raise ValueError("reviewed trace was not derived from the supplied run")
    return trace


def _apply_checkpoint(
    candidate: dict, task: dict, binding: dict, verdict: str, record: dict
) -> None:
    if (
        not candidate.get("checkpoint_review_required")
        or "checkpoint_review" in candidate
        or (
            task["checkpoint"] != candidate["prefix_text"]
            or binding["prefix_fingerprint"] != candidate["prefix_fingerprint"]
        )
    ):
        raise ValueError(f"Checkpoint binding changed: {record['review_id']}")
    if verdict not in {"correct", "incorrect", "unknown", "no_final_answer"}:
        raise ValueError(f"invalid Checkpoint verdict: {verdict}")
    candidate["checkpoint_review"] = {**record, "verdict": verdict}


def _apply_continuation(
    run_dir: Path, candidate: dict, task: dict, binding: dict, verdict: str, record: dict
) -> None:
    matches = [
        row
        for row in candidate["rollouts"]
        if (
            "logical_call_id" in binding
            and row.get("logical_call_id") == binding["logical_call_id"]
        )
        or ("logical_call_id" not in binding and row["rollout_index"] == binding["rollout_index"])
    ]
    if len(matches) != 1:
        raise ValueError(f"review does not bind to one frozen continuation: {record['review_id']}")
    rollout = matches[0]
    ref = rollout.get("response_ref")
    if rollout["status"] != "completed" or not ref or (binding.get("response_ref", ref) != ref):
        raise ValueError(f"continuation response identity changed: {record['review_id']}")
    payload = _read(run_dir / ref)["choices"][0]["message"]["content"]
    if payload != rollout["text"] or payload != task.get(
        "candidate_output", task.get("continuation")
    ):
        raise ValueError(
            f"blind task output does not match the saved response: {record['review_id']}"
        )
    if "checkpoint" in task and task["checkpoint"] != candidate["prefix_text"]:
        raise ValueError(f"blind task Checkpoint changed: {record['review_id']}")
    original = rollout["evaluation"]
    decision = AnswerDecision(
        AnswerStatus(original["status"]),
        original.get("extracted_answer"),
        original.get("normalized_answer"),
        original["rule_version"],
        original.get("reason"),
    )
    reviewed = apply_review_outcome(decision, verdict)
    original.update(
        record,
        prior_status=original["status"],
        status=reviewed.status.value,
        review_outcome=reviewed.review_outcome,
    )
