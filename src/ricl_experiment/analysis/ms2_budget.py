"""Reporting-only MS2 continuation costs and completion under observed token cutoffs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from ..evaluation.ms2_review import read_reviewed_trace
from .ms2 import route_flow_status


def continuation_budget_report(
    run_dir: Path, cutoffs: list[int], *, reviewed_trace: Path | None = None
) -> dict:
    """Read actual attempt usage and apply output cutoffs to saved whole responses."""
    if not cutoffs or any(value < 1 for value in cutoffs):
        raise ValueError("continuation output cutoffs must be positive")
    source = (run_dir / "traces/ms2.json").read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    trace = json.loads(source)
    if reviewed_trace is not None:
        trace = read_reviewed_trace(source, reviewed_trace)
    config = json.loads((run_dir / "config.json").read_bytes())
    observed_cap = config["ms2"]["continuation_max_tokens"]
    allowlist = {
        r["logical_call_id"]
        for sample in trace["samples"]
        for candidate in sample["candidates"]
        for r in candidate["rollouts"]
        if r.get("logical_call_id")
    }
    uri = (run_dir / "run.sqlite3").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        attempts = [
            dict(row)
            for row in db.execute(
                "SELECT logical_call_id, usage_status, input_tokens, output_tokens, reasoning_tokens, "
                "reserved_output_tokens, status FROM attempts"
            )
            if row["logical_call_id"] in allowlist
        ]
    known = [row for row in attempts if row["usage_status"] == "known"]
    unknown = [row for row in attempts if row["usage_status"] == "unknown"]

    observed = []
    for sample in trace["samples"]:
        for candidate in sample["candidates"]:
            for rollout in candidate["rollouts"]:
                ending = rollout.get("finish_reason")
                if (
                    ending is None
                    and rollout["status"] == "completed"
                    and rollout.get("response_ref")
                ):
                    saved = json.loads((run_dir / rollout["response_ref"]).read_bytes())
                    ending = saved["choices"][0].get("finish_reason")
                usage = rollout.get("usage")
                observed.append(
                    {
                        "sample_id": sample["sample_id"],
                        "route_id": candidate["route_id"],
                        "rollout_index": rollout["rollout_index"],
                        "route_flow": route_flow_status(candidate),
                        "outcome": _outcome(rollout, ending),
                        "finish_reason": ending,
                        "output_tokens": usage.get("output_tokens")
                        if isinstance(usage, Mapping)
                        else None,
                    }
                )

    def sensitivity(cap: int) -> dict:
        rows = Counter()
        for row in observed:
            outcome, tokens = row["outcome"], row["output_tokens"]
            if outcome in {"execution_failed", "not_dispatched"}:
                rows[outcome] += 1
            elif tokens is None:
                rows["usage_unknown"] += 1
            elif tokens > cap:
                rows["censored_above_cutoff"] += 1
            else:
                rows[outcome] += 1
        return {
            "output_cutoff": cap,
            "beyond_actual_call_cap": cap > observed_cap,
            "all_prescheduled_slots": len(observed),
            "outcomes": dict(rows),
        }

    return {
        "run_id": trace["run_id"],
        "phase": trace["phase"],
        "source_trace_sha256": digest,
        "reviewed_trace": reviewed_trace is not None,
        "actual_continuation_cap": observed_cap,
        "observed_outcomes": dict(Counter(row["outcome"] for row in observed)),
        "cost": {
            "attempts": len(attempts),
            "attempt_statuses": dict(Counter(a["status"] for a in attempts)),
            "known_usage_attempts": len(known),
            "unknown_usage_attempts": len(unknown),
            "known_input_tokens": sum(row["input_tokens"] for row in known),
            "known_output_tokens": sum(row["output_tokens"] for row in known),
            "reported_reasoning_tokens": sum(row["reasoning_tokens"] or 0 for row in known),
            "reasoning_usage_reported_attempts": sum(
                row["reasoning_tokens"] is not None for row in known
            ),
            "unknown_usage_output_reserve": sum(row["reserved_output_tokens"] for row in unknown),
        },
        "cutoffs": [sensitivity(cap) for cap in sorted(set(cutoffs))],
        "interpretation": (
            "Censoring uses complete observed response usage. It gives lower-bound observed successes "
            "retained at each cutoff, with no inference about rerunning at a different cap."
        ),
        "rollouts": observed,
    }


def _outcome(rollout: dict[str, Any], finish_reason: str | None) -> str:
    if rollout["status"] == "failed":
        return "execution_failed"
    if rollout["status"] != "completed":
        return "not_dispatched"
    grade = rollout.get("evaluation") or {}
    status = grade.get("status")
    if status in {"correct", "incorrect"}:
        return status
    if (
        finish_reason == "length"
        and grade.get("extracted_answer") is None
        and status
        in {
            "blind_review",
            "unknown",
            "no_evaluable_output",
        }
        and grade.get("reason")
        in {
            "non-empty output has no unambiguous extractable final answer",
            "no model output was produced",
        }
    ):
        return "incomplete_at_budget"
    if status == "no_evaluable_output":
        return "no_evaluable_output"
    return "grading_unknown"
