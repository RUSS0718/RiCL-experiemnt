"""Produce a coverage-only MS3-B report from the frozen run and blind review."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ricl_experiment.core.model import usage_out_of_bounds

GROUPS = ("G0", "G1", "G2", "G3")
COMPARISONS = (("G3", "G1"), ("G3", "G2"), ("G1", "G0"))
REVIEW_VALUES = {
    "repair_success": {"correct", "incorrect", "uncertain"},
    "error_removal": {"removed", "not_removed", "uncertain"},
    "new_error": {"present", "absent", "uncertain"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(
    run_dir: Path,
    manifest_path: Path,
    review_input_path: Path,
    review_path: Path,
    review_key_path: Path,
) -> dict[str, Any]:
    metadata = read_json(run_dir / "metadata.json")
    manifest = read_json(manifest_path)
    run_manifest = read_json(run_dir / "manifest.json")
    assignments = read_json(run_dir / "restricted" / "assignments.json")
    input_rows = read_json(review_input_path)["review_tasks"]
    review_rows = read_json(review_path)
    review_key = read_json(review_key_path)
    evaluations = [
        json.loads(line)
        for line in (run_dir / "blind" / "evaluations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if metadata.get("status") != "completed" or metadata.get("planned_logical_calls") != 64:
        raise ValueError("MS3-B run is not the completed frozen 64-call run")
    if len(manifest.get("samples", [])) != 16 or run_manifest.get("fingerprint") != manifest.get(
        "fingerprint"
    ):
        raise ValueError("MS3-B manifest does not match the completed run")
    if len(assignments) != 64 or len(evaluations) != 64:
        raise ValueError("expected exactly 64 assignments and evaluation records")
    by_eval = {row["evaluation_id"]: row for row in evaluations}
    if len(by_eval) != 64 or any(row["evaluation_id"] not in by_eval for row in assignments):
        raise ValueError("assignment/evaluation identities are incomplete or duplicated")

    input_ids = [row["review_id"] for row in input_rows]
    review_ids = [row["review_id"] for row in review_rows]
    if len(input_ids) != 45 or review_ids != input_ids or len(set(review_ids)) != 45:
        raise ValueError("blind review must cover the exact 45 exported tasks once")
    if set(review_key) != set(review_ids):
        raise ValueError("restricted blind-review key does not match the completed review")
    review_by_sample_group: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in review_rows:
        if any(row.get(field) not in allowed for field, allowed in REVIEW_VALUES.items()):
            raise ValueError("blind review contains an invalid or missing outcome value")
        key = review_key[row["review_id"]]
        sample_id, group = key["sample_id"], key["group"]
        if group not in GROUPS or group in review_by_sample_group[sample_id]:
            raise ValueError("restricted blind-review mapping is invalid")
        review_by_sample_group[sample_id][group] = row

    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for assignment in assignments:
        group, sample_id = assignment["group"], assignment["sample_id"]
        if group not in GROUPS or group in by_sample[sample_id]:
            raise ValueError("assignment sample/group mapping is invalid")
        evaluation = by_eval[assignment["evaluation_id"]]
        by_sample[sample_id][group] = {"assignment": assignment, "evaluation": evaluation}
    if len(by_sample) != 16 or any(set(rows) != set(GROUPS) for rows in by_sample.values()):
        raise ValueError("expected four arms for each of the 16 frozen samples")

    by_arm = {}
    for group in GROUPS:
        records = [rows[group] for rows in by_sample.values()]
        by_arm[group] = {
            "planned": 16,
            "nonempty_text_output": sum(
                row["assignment"]["status"] == "completed" for row in records
            ),
            "no_evaluable_output": sum(
                row["assignment"]["status"] == "no_evaluable_output" for row in records
            ),
        }

    comparison_coverage = {}
    for treated, control in COMPARISONS:
        raw_pairs = sum(
            by_sample[sample][treated]["assignment"]["status"] == "completed"
            and by_sample[sample][control]["assignment"]["status"] == "completed"
            for sample in by_sample
        )
        reviewed_pairs = sum(
            sample in review_by_sample_group
            and treated in review_by_sample_group[sample]
            and control in review_by_sample_group[sample]
            and review_by_sample_group[sample][treated]["repair_success"]
            in {"correct", "incorrect"}
            and review_by_sample_group[sample][control]["repair_success"]
            in {"correct", "incorrect"}
            for sample in by_sample
        )
        comparison_coverage[f"{treated}-{control}"] = {
            "nonempty_output_pairs": raw_pairs,
            "independently_reviewable_pairs": reviewed_pairs,
            "planned_pairs": 16,
        }
    common_four_arm = sum(
        all(
            group in review_by_sample_group.get(sample, {})
            and review_by_sample_group[sample][group]["repair_success"] in {"correct", "incorrect"}
            for group in GROUPS
        )
        for sample in by_sample
    )
    threshold = 13
    primary_gate_passed = all(
        result["independently_reviewable_pairs"] >= threshold
        for result in comparison_coverage.values()
    )

    empty_finish_reasons = Counter()
    missing_reasoning = 0
    usage_discrepancies = 0
    usage_discrepancies_without_evaluable_output = 0
    first_usage_discrepancy = None
    first_unreachable: dict[str, int] = {}
    unavailable: dict[str, set[str]] = defaultdict(set)
    for ordinal, assignment in enumerate(assignments, 1):
        attempts = assignment["attempts"]
        missing_reasoning += sum(attempt["reasoning_tokens"] is None for attempt in attempts)
        discrepancies = sum(usage_out_of_bounds(attempt) for attempt in attempts)
        usage_discrepancies += discrepancies
        if discrepancies and first_usage_discrepancy is None:
            first_usage_discrepancy = ordinal
        if assignment["status"] != "no_evaluable_output":
            continue
        usage_discrepancies_without_evaluable_output += discrepancies
        unavailable[assignment["group"]].add(assignment["sample_id"])
        for treated, control in COMPARISONS:
            if 16 - len(unavailable[treated] | unavailable[control]) < threshold:
                first_unreachable.setdefault(f"{treated}-{control}", ordinal)
        payload = read_json(run_dir / assignment["response_ref"])
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if not message.get("content"):
            empty_finish_reasons[str(choice.get("finish_reason") or "unknown")] += 1

    usage = metadata.get("usage", {})
    return {
        "status": "primary_gate_passed" if primary_gate_passed else "completed_coverage_only",
        "run_id": metadata["run_id"],
        "dataset": manifest.get("sources"),
        "manifest_fingerprint": manifest["fingerprint"],
        "sampling_seed": manifest.get("sampling", {}).get("seed"),
        "planned_groups": 16,
        "planned_sends": 64,
        "actual_sends": usage.get("attempts"),
        "unknown_usage_attempts": usage.get("unknown_usage_attempts"),
        "unknown_reasoning_attempts": missing_reasoning,
        "usage_discrepancies": usage_discrepancies,
        "usage_discrepancies_without_evaluable_output": (
            usage_discrepancies_without_evaluable_output
        ),
        "first_usage_discrepancy_send": first_usage_discrepancy,
        "first_unreachable_pair_gate_send": first_unreachable,
        "usage_tokens": {
            "input": usage.get("input_tokens"),
            "output": usage.get("output_tokens"),
            "reasoning_reported": usage.get("reasoning_tokens"),
        },
        "actual_cost_usd": None,
        "cost_status": "unknown_not_zero; provider did not return actual charge",
        "arm_output_coverage": by_arm,
        "comparison_coverage": comparison_coverage,
        "four_arm_common_reviewable_groups": common_four_arm,
        "primary_pair_threshold": threshold,
        "primary_gate_passed": primary_gate_passed,
        "reporting_mode": "paired coverage only; no arm-effect estimates"
        if not primary_gate_passed
        else "coverage recorded; effects require separate analysis",
        "blind_review": {
            "reviewed_nonempty_outputs": len(review_rows),
            "reviewer_kind": "Agent-Assessed",
            "human_signoff": False,
            "uncertain_repair_judgments": sum(
                row["repair_success"] == "uncertain" for row in review_rows
            ),
        },
        "empty_content_finish_reasons": dict(empty_finish_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-input", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    result = summarize(args.run_dir, args.manifest, args.review_input, args.review, args.review_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
