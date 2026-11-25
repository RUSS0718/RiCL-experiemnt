"""Controlled-fixture contracts for offline MS1 dependency analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from ricl_experiment.motivation.ms1_dependency import build_ms1_sampling_plan


def _candidate(sample_id: str, status: str, index: int, *, group: str) -> dict:
    prefix = f"Checkpoint for {sample_id}."
    continuation = f"Continue {sample_id}; final answer: {status}."
    return {
        "candidate_index": index,
        "candidate_id": f"candidate-{sample_id}-{index}",
        "route_id": f"route-{sample_id}-{index}",
        "problem_id": sample_id,
        "checkpoint_id": f"checkpoint-{sample_id}-{index}",
        "prefix_text": prefix,
        "prefix_fingerprint": f"prefix-{sample_id}-{index}",
        "prefix_status": "available",
        "truncation_status": "not_truncated",
        "stop_reason": "stop",
        "last_step_status": "complete",
        "initial_evaluation": {"status": "unknown"},
        "rollouts": [
            {
                "rollout_index": 0,
                "status": "completed",
                "checkpoint_prefix_fingerprint": f"prefix-{sample_id}-{index}",
                "logical_call_id": f"call-{sample_id}-{index}",
                "text": continuation,
                "evaluation": {"status": status},
                "usage": {"status": "known", "output_tokens": 5},
            },
            {
                "rollout_index": 1,
                "status": "completed",
                "checkpoint_prefix_fingerprint": f"prefix-{sample_id}-{index}",
                "logical_call_id": f"call-{sample_id}-{index}-1",
                "text": continuation,
                "evaluation": {"status": status},
                "usage": {"status": "known", "output_tokens": 5},
            },
        ],
        "segments": [
            {
                "kind": "checkpoint_prefix",
                "status": "completed",
                "start": 0,
                "end": len(prefix),
                "text": prefix,
            },
            {
                "kind": "rollout_0",
                "status": "completed",
                "start": len(prefix),
                "end": len(prefix) + len(continuation),
                "text": continuation,
            },
            {
                "kind": "rollout_1",
                "status": "completed",
                "start": len(prefix),
                "end": len(prefix) + len(continuation),
                "text": continuation,
            },
        ],
        "question_group_id_for_fixture": group,
    }


def _trace(rows: list[tuple[str, str, str, str]]) -> dict:
    samples = []
    for sample_id, source, group, status in rows:
        samples.append(
            {
                "sample_id": sample_id,
                "problem": f"Solve {sample_id}.",
                "source": source,
                "question_group_id": group,
                "candidates": [_candidate(sample_id, status, 0, group=group)],
            }
        )
    return {
        "schema_version": 1,
        "run_id": "fixture-run",
        "study": "ms2",
        "phase": "pilot",
        "manifest_id": "fixture-manifest",
        "manifest_fingerprint": "fixture-fingerprint",
        "candidate_slots": 3,
        "rollouts_per_checkpoint": 2,
        "continuation_template_version": "ms2-neutral-continuation-v1",
        "samples": samples,
        "usage": {},
    }


def _independent_outcomes(trace: dict) -> list[dict]:
    records = []
    for sample in trace["samples"]:
        for candidate in sample.get("candidates", []):
            for rollout_index, rollout in enumerate(candidate.get("rollouts", [])):
                records.append(
                    {
                        "path_id": f"{trace['run_id']}:{sample['sample_id']}:{candidate['candidate_id']}:rollout:{rollout_index}",
                        "status": rollout["evaluation"]["status"],
                        "checkpoint_status": candidate["initial_evaluation"]["status"],
                        "reference": f"fixture-outcome:{sample['sample_id']}:{rollout_index}",
                    }
                )
    return records


def test_sampling_is_fixed_seed_stratified_and_group_limited():
    traces = [
        _trace(
            [
                ("f1", "math500", "shared", "incorrect"),
                ("f2", "math500", "failure-2", "incorrect"),
                ("f3", "omni_math", "failure-3", "incorrect"),
                ("c1", "math500", "shared", "correct"),
                ("c2", "omni_math", "correct-2", "correct"),
                ("c3", "omni_math", "correct-3", "correct"),
            ]
        )
    ]

    first = build_ms1_sampling_plan(
        traces,
        seed=17,
        independent_outcomes=_independent_outcomes(traces[0]),
        failure_target=2,
        correct_reference_target=2,
    )
    second = build_ms1_sampling_plan(
        deepcopy(traces),
        seed=17,
        independent_outcomes=_independent_outcomes(traces[0]),
        failure_target=2,
        correct_reference_target=2,
    )

    assert first == second
    assert [row["sampling_outcome"] for row in first["selected_paths"]].count("incorrect") == 2
    assert [row["sampling_outcome"] for row in first["selected_paths"]].count("correct") == 2
    assert len({row["question_group_id"] for row in first["selected_paths"]}) == 4
    assert first["strata"]["math500|incorrect"]["candidate_count"] == 4
    assert first["strata"]["omni_math|correct"]["selected_count"] >= 1
    assert first["deficits"] == {"incorrect": 0, "correct": 0}


def test_sampling_reports_generation_failures_separately_from_unsegmentable_paths():
    trace = _trace([("failed", "math500", "failed-group", "incorrect")])
    candidate = trace["samples"][0]["candidates"][0]
    candidate["generation_status"] = "failed"
    candidate["generation_failure"] = "fixture generation failure"
    candidate["prefix_status"] = "unavailable"
    plan = build_ms1_sampling_plan(
        [trace],
        seed=0,
        independent_outcomes=_independent_outcomes(trace),
        failure_target=1,
        correct_reference_target=1,
    )
    assert plan["selected_paths"] == []
    assert any(row["reason"] == "runtime_failure" for row in plan["excluded"])


def test_sampling_reports_ineligible_records_and_quota_gaps():
    trace = _trace(
        [
            ("f1", "math500", "failure-1", "incorrect"),
            ("c1", "math500", "correct-1", "correct"),
        ]
    )
    trace["samples"].append(
        {
            "sample_id": "unknown",
            "problem": "Solve unknown.",
            "source": "math500",
            "question_group_id": "unknown-group",
            "candidates": [],
        }
    )

    plan = build_ms1_sampling_plan(
        [trace],
        seed=0,
        independent_outcomes=_independent_outcomes(trace),
        failure_target=2,
        correct_reference_target=2,
    )

    assert plan["deficits"] == {"incorrect": 1, "correct": 1}
    assert any(row["reason"] == "no_usable_rollout" for row in plan["excluded"])
    assert plan["denominators"]["candidate_paths"] == 4


def _review(path_id: str, reviewer_id: str, *, second: bool = False) -> dict:
    return {
        "path_id": path_id,
        "reviewer_id": reviewer_id,
        "nodes": [
            {
                "step_id": "n0",
                "span": {"start": 0, "end": 8},
                "claim": "x=2 or x=-2",
                "is_nontrivial": True,
                "applicability": "x is real",
                "dependency_ids": [],
                "validity_status": "Valid",
                "verification_evidence": "follows from x^2=4",
                "token_count": 5,
                "equivalence_group_id": "g0",
            },
            {
                "step_id": "n1",
                "span": {"start": 9, "end": 16},
                "claim": "x=-2",
                "is_nontrivial": True,
                "applicability": "x>0",
                "dependency_ids": ["n0"],
                "dependency_status": {"n0": "confirmed"},
                "validity_status": "Unknown" if second else "Invalid",
                "verification_evidence": "sign condition is inconsistent",
                "token_count": 3,
                "equivalence_group_id": "g1",
            },
        ],
        "first_error_step_id": "n1",
        "first_error_status": "confirmed",
    }


def test_blind_annotation_import_reconciles_with_a_third_review():
    from ricl_experiment.motivation.ms1_dependency import (
        build_blind_annotation_manifest,
        import_ms1_annotations,
    )

    trace = _trace(
        [("f1", "math500", "failure-1", "incorrect"), ("c1", "math500", "correct-1", "correct")]
    )
    plan = build_ms1_sampling_plan(
        [trace],
        seed=3,
        independent_outcomes=_independent_outcomes(trace),
        failure_target=1,
        correct_reference_target=1,
    )
    blind = build_blind_annotation_manifest(plan)
    assert blind["paths"]
    assert all("sampling_outcome" not in path for path in blind["paths"])
    assert all("evaluation" not in path for path in blind["paths"])

    path_id = blind["paths"][0]["path_id"]
    independent = []
    third = []
    for index, path in enumerate(blind["paths"]):
        current = path["path_id"]
        independent.extend(
            [_review(current, f"r1-{index}"), _review(current, f"r2-{index}", second=True)]
        )
        if current == path_id:
            third.append(_review(current, "r3"))
    imported = import_ms1_annotations(plan, independent, third)
    result = next(row for row in imported["paths"] if row["path_id"] == path_id)
    assert result["reconciliation_status"] == "third_reviewer_reconciled"
    assert result["consensus"]["nodes"][1]["validity_status"] == "Invalid"
    assert result["third_review"]["role"] == "third_reviewer"


def test_annotation_import_keeps_unresolved_disagreement_unknown():
    from ricl_experiment.motivation.ms1_dependency import import_ms1_annotations

    plan = {
        "seed": 1,
        "selected_paths": [{"path_id": "p", "sample_id": "s", "question_group_id": "g"}],
    }
    imported = import_ms1_annotations(plan, [_review("p", "r1"), _review("p", "r2", second=True)])
    result = imported["paths"][0]
    assert result["reconciliation_status"] == "pending_third_review"
    assert result["consensus"]["first_error_status"] == "unknown"
    assert result["consensus"]["nodes"][1]["validity_status"] == "Unknown"


def test_dependency_metrics_follow_claim_reuse_and_propagation_denominators():
    from ricl_experiment.analysis.ms1_dependency import analyze_ms1_dependency

    nodes = [
        {
            "step_id": "n0",
            "claim": "x in {-2,2}",
            "validity_status": "Valid",
            "verification_evidence": "square roots",
            "equivalence_group_id": "g0",
            "dependency_ids": [],
        },
        {
            "step_id": "n1",
            "claim": "x=-2",
            "validity_status": "Invalid",
            "verification_evidence": "violates x>0",
            "equivalence_group_id": "g1",
            "dependency_ids": ["n0"],
            "dependency_status": {"n0": "confirmed"},
        },
        {
            "step_id": "n2",
            "claim": "x^4=16",
            "validity_status": "Unsupported",
            "verification_evidence": "true expression from invalid sign choice",
            "equivalence_group_id": "g2",
            "dependency_ids": ["n1"],
            "dependency_status": {"n1": "confirmed"},
        },
        {
            "step_id": "n3",
            "claim": "x^8=256",
            "validity_status": "Unsupported",
            "verification_evidence": "true expression from unsupported step",
            "equivalence_group_id": "g3",
            "dependency_ids": ["n2"],
            "dependency_status": {"n2": "confirmed"},
        },
        {
            "step_id": "n4",
            "claim": "x=2",
            "validity_status": "Valid",
            "verification_evidence": "positive root",
            "equivalence_group_id": "g4",
            "dependency_ids": ["n0"],
            "dependency_status": {"n0": "confirmed"},
        },
        {
            "step_id": "n5",
            "claim": "x^4=16",
            "validity_status": "Valid",
            "verification_evidence": "square the given condition",
            "equivalence_group_id": "g2",
            "dependency_ids": [],
        },
    ]

    report = analyze_ms1_dependency(
        [
            {
                "path_id": "p",
                "first_error_step_id": "n1",
                "first_error_status": "confirmed",
                "nodes": nodes,
            }
        ],
        [
            {
                "path_id": "p",
                "status": "completed",
                "token_spans": [[4, 8], [6, 10]],
                "visible_output_tokens": 20,
                "hidden_usage_tokens": 7,
            }
        ],
    )
    path = report["paths"][0]

    assert path["error_propagation_ratio"] == {
        "numerator": 2,
        "denominator": 4,
        "value": 0.5,
        "unresolved_dependency_edges": 0,
        "not_applicable_reason": None,
    }
    assert path["dependency_depth"]["by_step"] == {"n2": 1, "n3": 2}
    assert path["reusable_claim_ratio"] == {
        "numerator": 3,
        "denominator": 5,
        "value": 0.6,
        "not_applicable_reason": None,
    }
    assert path["rederivation"] == {
        "status": "completed",
        "rederived_claim_tokens": 6,
        "visible_output_tokens": 20,
        "rederived_claim_ratio": 0.3,
        "token_count_estimated": False,
        "hidden_usage": {"status": "known", "tokens": 7},
        "matched_span_count": 2,
    }
    assert report["summary"]["error_propagation_ratio"]["denominator"] == 4


def test_rederivation_uses_estimate_flag_and_keeps_unknown_hidden_usage_separate():
    from ricl_experiment.analysis.ms1_dependency import analyze_ms1_dependency

    report = analyze_ms1_dependency(
        [{"path_id": "p", "first_error_status": "unknown", "nodes": []}],
        [
            {
                "path_id": "p",
                "status": "completed",
                "visible_text": "x = 2; therefore x is positive",
                "character_spans": [[0, 5]],
            }
        ],
    )
    rederived = report["paths"][0]["rederivation"]
    assert rederived["token_count_estimated"] is True
    assert rederived["hidden_usage"] == {"status": "unknown", "tokens": None}
    assert report["paths"][0]["error_propagation_ratio"]["value"] is None


def test_metrics_preserve_non_applicability_and_ignore_excluded_edges_for_reuse():
    from ricl_experiment.analysis.ms1_dependency import analyze_ms1_dependency

    report = analyze_ms1_dependency(
        [
            {
                "path_id": "missing-anchor",
                "first_error_step_id": "absent",
                "first_error_status": "confirmed",
                "nodes": [{"step_id": "n1", "claim": "x=1", "validity_status": "Valid"}],
            },
            {
                "path_id": "excluded-edge",
                "first_error_status": "unknown",
                "nodes": [
                    {
                        "step_id": "bad",
                        "claim": "x=0",
                        "validity_status": "Invalid",
                        "verification_evidence": "contradicts the premise",
                    },
                    {
                        "step_id": "independent",
                        "claim": "x=1",
                        "validity_status": "Valid",
                        "verification_evidence": "direct substitution",
                        "dependency_ids": ["bad"],
                        "dependency_status": {"bad": "excluded"},
                    },
                ],
            },
        ]
    )

    missing = report["paths"][0]["error_propagation_ratio"]
    assert missing["value"] is None
    assert missing["not_applicable_reason"] == "first_error_anchor_missing"
    assert report["paths"][1]["reusable_claim_ratio"]["numerator"] == 1


def test_reusable_claim_ratio_rejects_transitively_unknown_support():
    from ricl_experiment.analysis.ms1_dependency import analyze_ms1_dependency

    report = analyze_ms1_dependency(
        [
            {
                "path_id": "p",
                "first_error_status": "unknown",
                "nodes": [
                    {
                        "step_id": "premise",
                        "claim": "x=1",
                        "validity_status": "Valid",
                        "verification_evidence": "direct",
                        "dependency_ids": ["missing"],
                        "dependency_status": {"missing": "unknown"},
                    },
                    {
                        "step_id": "claim",
                        "claim": "x=2",
                        "validity_status": "Valid",
                        "verification_evidence": "uses premise",
                        "dependency_ids": ["premise"],
                    },
                ],
            }
        ]
    )
    assert report["paths"][0]["reusable_claim_ratio"]["numerator"] == 0


def test_reusable_claim_ratio_excludes_explicitly_trivial_nodes():
    from ricl_experiment.analysis.ms1_dependency import analyze_ms1_dependency

    report = analyze_ms1_dependency(
        [
            {
                "path_id": "p",
                "first_error_status": "unknown",
                "nodes": [
                    {
                        "step_id": "trivial",
                        "claim": "given",
                        "is_nontrivial": False,
                        "validity_status": "Valid",
                    },
                    {
                        "step_id": "claim",
                        "claim": "x=2",
                        "is_nontrivial": True,
                        "validity_status": "Valid",
                        "verification_evidence": "follows from the premise",
                    },
                ],
            }
        ]
    )
    assert report["paths"][0]["reusable_claim_ratio"]["numerator"] == 1
    assert report["paths"][0]["reusable_claim_ratio"]["denominator"] == 1


def test_case_export_supports_two_or_three_selected_paths_only(tmp_path):
    from ricl_experiment.analysis.ms1_dependency import (
        analyze_ms1_dependency,
        write_ms1_case_export,
    )

    annotations = [
        {"path_id": path_id, "first_error_status": "unknown", "nodes": []}
        for path_id in ("p1", "p2", "p3")
    ]
    report = analyze_ms1_dependency(annotations)
    output = tmp_path / "cases.json"
    write_ms1_case_export(report, ["p1", "p2"], output)
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["case_count"] == 2
    assert [row["path_id"] for row in exported["cases"]] == ["p1", "p2"]

    with pytest.raises(ValueError, match="2 or 3"):
        write_ms1_case_export(report, ["p1"], tmp_path / "one.json")


def test_sampling_uses_external_independent_outcomes_not_trace_answer_heuristic():
    trace = _trace([("sample", "math500", "group", "correct")])
    outcomes = _independent_outcomes(trace)
    outcomes[0]["status"] = "incorrect"
    outcomes[0]["reference"] = "review-ledger:case-1"
    outcomes[1]["status"] = "unknown"
    outcomes[1]["reference"] = "review-ledger:case-2"

    plan = build_ms1_sampling_plan(
        [trace],
        seed=12,
        independent_outcomes=outcomes,
        failure_target=1,
        correct_reference_target=1,
    )

    assert len(plan["selected_paths"]) == 1
    selected = plan["selected_paths"][0]
    assert selected["rollout_index"] == 0
    assert selected["sampling_outcome"] == "incorrect"
    assert selected["outcome_reference"] == "review-ledger:case-1"
    assert plan["deficits"] == {"incorrect": 0, "correct": 1}
    assert any(row["reason"] == "independent_outcome_unknown" for row in plan["excluded"])


def test_sampling_excludes_outcomes_without_a_review_reference():
    trace = _trace([("sample", "math500", "group", "incorrect")])
    outcomes = _independent_outcomes(trace)
    outcomes[0].pop("reference")
    plan = build_ms1_sampling_plan(
        [trace], seed=0, independent_outcomes=outcomes, failure_target=1, correct_reference_target=1
    )
    assert len(plan["selected_paths"]) == 1
    assert plan["selected_paths"][0]["rollout_index"] == 1
    assert plan["deficits"] == {"incorrect": 0, "correct": 1}
    assert any(
        row.get("rollout_index") == 0 and row["reason"] == "independent_outcome_reference_missing"
        for row in plan["excluded"]
    )


def test_sampling_fills_available_quota_without_requiring_all_targets():
    trace = _trace([("only-group", "math500", "one-group", "incorrect")])
    plan = build_ms1_sampling_plan(
        [trace],
        seed=0,
        independent_outcomes=_independent_outcomes(trace),
        failure_target=1,
        correct_reference_target=1,
    )
    assert len(plan["selected_paths"]) == 1
    assert plan["selected_paths"][0]["question_group_id"] == "one-group"
    assert plan["deficits"] == {"incorrect": 0, "correct": 1}


def test_sampling_never_selects_one_question_group_in_both_outcomes():
    trace = _trace([("shared-group", "math500", "same-question", "incorrect")])
    outcomes = _independent_outcomes(trace)
    outcomes[1]["status"] = "correct"

    plan = build_ms1_sampling_plan(
        [trace],
        seed=23,
        independent_outcomes=outcomes,
        failure_target=1,
        correct_reference_target=1,
    )

    assert len(plan["selected_paths"]) == 1
    assert plan["selected_paths"][0]["sampling_outcome"] == "incorrect"
    assert plan["selected_paths"][0]["question_group_id"] == "same-question"
    assert plan["deficits"] == {"incorrect": 0, "correct": 1}


def test_sampling_excludes_early_correct_checkpoints():
    trace = _trace([("sample", "math500", "group", "incorrect")])
    trace["samples"][0]["candidates"][0]["initial_evaluation"]["status"] = "correct"
    outcomes = _independent_outcomes(trace)
    outcomes[0].pop("checkpoint_status")
    plan = build_ms1_sampling_plan(
        [trace], seed=0, independent_outcomes=outcomes, failure_target=1, correct_reference_target=1
    )
    assert all(row["rollout_index"] != 0 for row in plan["selected_paths"])
    assert any(
        row.get("rollout_index") == 0 and row["reason"] == "checkpoint_completed_correct"
        for row in plan["excluded"]
    )


def test_sampling_rejects_missing_independent_outcome_sidecar():
    with pytest.raises(ValueError, match="independent_outcomes sidecar is required"):
        build_ms1_sampling_plan([_trace([("s", "math500", "g", "correct")])], seed=0)


def test_sampling_keeps_only_the_selected_continuation_and_excludes_unsegmentable_paths():
    from ricl_experiment.motivation.ms1_dependency import build_blind_annotation_manifest

    trace = _trace([("sample", "math500", "group", "incorrect")])
    candidate = trace["samples"][0]["candidates"][0]
    candidate["rollouts"][1]["text"] = "sibling continuation"
    candidate["segments"][2]["end"] = len(candidate["prefix_text"]) + len("sibling continuation")
    candidate["segments"][2]["text"] = "sibling continuation"
    outcomes = _independent_outcomes(trace)
    plan = build_ms1_sampling_plan(
        [trace], seed=0, independent_outcomes=outcomes, failure_target=1, correct_reference_target=1
    )
    blind = build_blind_annotation_manifest(plan)
    selected = blind["paths"][0]
    assert [segment["kind"] for segment in selected["segments"]] == [
        "checkpoint_prefix",
        f"rollout_{selected['rollout_index']}",
    ]

    unsegmentable = deepcopy(trace)
    unsegmentable["samples"][0]["candidates"][0]["prefix_status"] = "unavailable"
    excluded_plan = build_ms1_sampling_plan(
        [unsegmentable],
        seed=0,
        independent_outcomes=outcomes,
        failure_target=1,
        correct_reference_target=1,
    )
    assert excluded_plan["selected_paths"] == []
    assert any(row["reason"] == "unsegmentable_path" for row in excluded_plan["excluded"])


def test_ms1_cli_completes_offline_fixture_pipeline(tmp_path):
    trace = _trace(
        [("f1", "math500", "failure-1", "incorrect"), ("c1", "math500", "correct-1", "correct")]
    )
    trace_path = tmp_path / "ms2-trace.json"
    outcomes_path = tmp_path / "outcomes.json"
    plan_path = tmp_path / "plan.json"
    blind_path = tmp_path / "blind.json"
    annotation_path = tmp_path / "annotations.json"
    analysis_path = tmp_path / "analysis.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    outcomes_path.write_text(json.dumps(_independent_outcomes(trace)), encoding="utf-8")

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "ricl_experiment", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    result = run(
        "ms1-plan",
        "--trace",
        str(trace_path),
        "--outcomes",
        str(outcomes_path),
        "--seed",
        "3",
        "--failure-target",
        "1",
        "--correct-target",
        "1",
        "--output",
        str(plan_path),
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    result = run("ms1-blind", "--plan", str(plan_path), "--output", str(blind_path))
    assert result.returncode == 0, result.stderr
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    independent = []
    for index, path in enumerate(blind["paths"]):
        independent.append(_review(path["path_id"], f"r1-{index}"))
    independent_path = tmp_path / "independent.json"
    independent_path.write_text(json.dumps(independent), encoding="utf-8")
    second_path = tmp_path / "independent-2.json"
    second_path.write_text(
        json.dumps(
            [_review(path["path_id"], f"r2-{index}") for index, path in enumerate(blind["paths"])]
        ),
        encoding="utf-8",
    )

    result = run(
        "ms1-import-annotations",
        "--plan",
        str(plan_path),
        "--independent",
        str(independent_path),
        "--independent",
        str(second_path),
        "--output",
        str(annotation_path),
    )
    assert result.returncode == 0, result.stderr

    result = run(
        "ms1-analyze",
        "--annotations",
        str(annotation_path),
        "--output",
        str(analysis_path),
    )
    assert result.returncode == 0, result.stderr
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["denominators"]["annotation_paths"] == len(plan["selected_paths"])


def test_ms1_cli_reads_jsonl_rederivation_records(tmp_path):
    annotations_path = tmp_path / "annotations.json"
    rederivations_path = tmp_path / "rederivations.jsonl"
    output_path = tmp_path / "analysis.json"
    annotations_path.write_text(
        json.dumps({"paths": [{"path_id": "p", "first_error_status": "unknown", "nodes": []}]}),
        encoding="utf-8",
    )
    rederivations_path.write_text(
        json.dumps(
            {
                "path_id": "p",
                "status": "completed",
                "token_spans": [[0, 2]],
                "visible_output_tokens": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ricl_experiment",
            "ms1-analyze",
            "--annotations",
            str(annotations_path),
            "--rederivations",
            str(rederivations_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["denominators"]["rederivation_records"]
        == 1
    )
