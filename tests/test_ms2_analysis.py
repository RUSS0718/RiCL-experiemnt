from __future__ import annotations

import json
from copy import deepcopy

import pytest

from ricl_experiment.analysis.ms2 import analyze_ms2_trace, write_ms2_analysis


def _candidate(candidate_id: str, y: tuple[str, str], rater: float, progress: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "route_id": candidate_id,
        "prefix_status": "available",
        "initial_evaluation": {"analysis_category": "early_incorrect"},
        "rollouts": [{"status": "completed", "evaluation": {"status": value}} for value in y],
        "scores": {
            "rater": rater,
            "verified_progress": progress,
        },
    }


def _trace() -> dict:
    return {
        "schema_version": 1,
        "study": "ms2",
        "phase": "confirmatory",
        "run_id": "run-1",
        "manifest_id": "manifest-1",
        "manifest_fingerprint": "digest-1",
        "candidate_slots": 3,
        "rollouts_per_checkpoint": 2,
        "samples": [
            {
                "sample_id": "q1-a",
                "source": "math500",
                "question_group_id": "group-1",
                "candidates": [
                    _candidate("a", ("correct", "correct"), 2, 3),
                    _candidate("b", ("correct", "incorrect"), 3, 2),
                    _candidate("c", ("incorrect", "incorrect"), 1, 1),
                ],
            }
        ],
    }


def test_main_comparisons_match_frozen_selection_and_ranking_example():
    report = analyze_ms2_trace(_trace(), bootstrap_replicates=100, seed=7)
    assert report["summary"]["selection"]["rater"]["value"] == 0.5
    assert report["summary"]["selection"]["verified_progress"]["value"] == 1.0
    assert report["summary"]["selection"]["rater_plus_evidence"]["value"] == 0.75
    assert report["summary"]["selection"]["random"]["value"] == 0.5
    assert report["summary"]["ranking"]["rater"]["value"] == 2 / 3
    assert report["summary"]["ranking"]["rater_plus_evidence"]["value"] == 5 / 6
    assert report["summary"]["ranking"]["verified_progress"]["value"] == 1.0
    assert report["summary"]["ranking"]["random"]["value"] == 0.5
    assert report["comparisons"]["selection"]["rater_plus_evidence_minus_rater"]["estimate"] == 0.25
    assert report["comparisons"]["ranking"]["rater_plus_evidence_minus_rater"][
        "estimate"
    ] == pytest.approx(1 / 6)
    assert report["coverage"]["selection_samples"] == 1
    assert report["coverage"]["ranking_samples"] == 1


def test_analysis_keeps_missing_routes_and_excludes_them_from_main_denominators():
    trace = _trace()
    trace["samples"][0]["candidates"][1]["rollouts"][1]["evaluation"]["status"] = "unknown"
    report = analyze_ms2_trace(trace, bootstrap_replicates=10, seed=0)
    assert report["coverage"]["selection_samples"] == 0
    assert report["coverage"]["ranking_samples"] == 0
    assert report["exclusions"]["q1-a"]["reasons"] == ["route_missing_independent_evaluation"]
    assert report["diagnostics"]["missing_rollout_evaluations"] == 1


def test_cluster_bootstrap_reports_unestimable_single_group_interval():
    report = analyze_ms2_trace(_trace(), bootstrap_replicates=100, seed=7)
    interval = report["comparisons"]["selection"]["rater_plus_evidence_minus_rater"]["ci_95"]
    assert interval is None
    assert report["comparisons"]["selection"]["rater_plus_evidence_minus_rater"]["ci_reason"]


def test_rater_plus_evidence_averages_shared_candidate_ranks_not_raw_scores():
    trace = _trace()
    candidates = trace["samples"][0]["candidates"]
    candidates[0]["scores"] = {"rater": 3, "verified_progress": 0}
    candidates[1]["scores"] = {"rater": 2, "verified_progress": 2}
    candidates[2]["scores"] = {"rater": 1, "verified_progress": 3}

    report = analyze_ms2_trace(trace, bootstrap_replicates=0)

    assert report["summary"]["selection"]["rater_plus_evidence"]["value"] == 0.5
    assert report["summary"]["ranking"]["rater_plus_evidence"]["value"] == 0.5


def test_early_completion_routes_are_reported_but_removed_before_common_candidate_analysis():
    trace = _trace()
    first = trace["samples"][0]
    first["candidates"][0]["initial_evaluation"] = {
        "status": "correct",
        "analysis_category": "early_correct",
        "extracted_answer": "42",
    }
    first["candidates"][0]["rollouts"][0]["status"] = "failed"
    first["candidates"][1]["rollouts"][1]["evaluation"]["status"] = "correct"
    first["candidates"][1]["rollouts"][0]["evaluation"]["status"] = "correct"
    second = deepcopy(first)
    second["sample_id"] = "q1-b"
    second["question_group_id"] = "group-2"
    second["candidates"][0]["initial_evaluation"] = {
        "status": "unknown",
        "analysis_category": "early_unknown",
        "extracted_answer": "ambiguous answer",
    }
    trace["samples"].append(second)

    report = analyze_ms2_trace(trace, bootstrap_replicates=0)

    assert report["coverage"]["selection_samples"] == 2
    assert report["diagnostics"]["early_correct_routes"] == 1
    assert report["diagnostics"]["early_unknown_routes"] == 1
    assert report["per_sample"][0]["status"] == "included"
    assert report["per_sample"][0]["eligible_route_count"] == 2
    assert report["per_sample"][0]["early_correct_route_count"] == 1
    assert report["per_sample"][1]["early_unknown_route_count"] == 1


def test_analysis_retains_per_sample_exclusions_and_uses_available_replicates_per_group():
    trace = _trace()
    valid = trace["samples"][0]
    failed = {
        **valid,
        "sample_id": "q1-b",
        "candidates": deepcopy(valid["candidates"]),
    }
    failed["candidates"][1]["rollouts"][0]["status"] = "failed"
    trace["samples"].append(failed)

    report = analyze_ms2_trace(trace, bootstrap_replicates=0)

    assert report["per_sample"][1]["status"] == "excluded"
    assert report["per_sample"][1]["reasons"] == ["route_missing_independent_evaluation"]
    assert report["summary"]["selection"]["rater"]["value"] == 0.5
    assert report["summary"]["selection"]["rater"]["question_group_count"] == 1
    assert report["coverage"]["analyzable_samples"] == 1
    assert report["diagnostics"]["missing_rollout_evaluations"] == 1


def test_legal_zero_scores_are_analyzed_and_evidence_ties_are_counted():
    trace = _trace()
    for candidate in trace["samples"][0]["candidates"]:
        candidate["scores"] = {"rater": 0, "verified_progress": 0}

    report = analyze_ms2_trace(trace, bootstrap_replicates=0)

    assert report["coverage"]["selection_samples"] == 1
    assert report["summary"]["selection"]["verified_progress"]["value"] == 0.5
    assert report["evidence_coverage"]["score_distribution"] == {"0": 3}
    assert report["evidence_coverage"]["tied_pairs"] == 3
    assert report["evidence_coverage"]["eligible_pairs"] == 3
    assert report["evidence_coverage"]["all_tied_sample_count"] == 1


def test_score_missing_rate_uses_each_scoring_method_slot(tmp_path):
    trace = _trace()
    scoring_dir = tmp_path / "scoring"
    scoring_dir.mkdir()
    for candidate in trace["samples"][0]["candidates"]:
        candidate.pop("scores")
        (scoring_dir / f"{candidate['route_id']}.json").write_text(
            json.dumps(
                {
                    "sample_id": "q1-a",
                    "route_id": candidate["route_id"],
                    "llm_rater": {"score": 2, "parse_status": "valid"},
                    "verified_progress": {"score": None, "coverage_status": "missing"},
                }
            ),
            encoding="utf-8",
        )

    report = analyze_ms2_trace(trace, scoring_dir=scoring_dir, bootstrap_replicates=0)

    assert report["coverage"]["expected_score_slots"] == 6
    assert report["coverage"]["valid_score_slots"] == 3
    assert report["coverage"]["missing_score_slots"] == 3
    assert report["coverage"]["score_missing_rate"] == 0.5
    assert report["coverage"]["score_slots_by_method"] == {
        "rater": {"expected": 3, "valid": 3, "missing": 0, "missing_rate": 0.0},
        "verified_progress": {
            "expected": 3,
            "valid": 0,
            "missing": 3,
            "missing_rate": 1.0,
        },
    }
    assert report["evidence_coverage"]["coverage_status_counts"] == {"missing": 3}


def test_analysis_rejects_duplicate_route_identity_within_a_sample():
    trace = _trace()
    trace["samples"][0]["candidates"][1]["route_id"] = "a"

    with pytest.raises(ValueError, match="route IDs must be unique within a sample"):
        analyze_ms2_trace(trace, bootstrap_replicates=0)

def test_writer_loads_scoring_artifacts_and_writes_offline_coverage_figure(tmp_path):
    run_dir = tmp_path / "run"
    trace_path = run_dir / "traces" / "ms2.json"
    scoring_dir = run_dir / "scoring"
    trace_path.parent.mkdir(parents=True)
    scoring_dir.mkdir()
    trace = _trace()
    for candidate in trace["samples"][0]["candidates"]:
        candidate.pop("scores")
        for rollout in candidate["rollouts"]:
            rollout["usage"] = {
                "status": "known",
                "input_tokens": 4,
                "output_tokens": 5,
                "reasoning_tokens": 0,
            }
        (scoring_dir / f"{candidate['route_id']}.json").write_text(
            json.dumps(
                {
                    "sample_id": "q1-a",
                    "route_id": candidate["route_id"],
                    "llm_rater": {"score": 0, "parse_status": "valid"},
                    "verified_progress": {
                        "score": 0,
                        "coverage": 0,
                        "coverage_status": "unverified",
                        "verified": 0,
                        "falsified": 0,
                        "unknown": 1,
                        "evidence_by_source": {
                            "Unknown": {
                                "claims": 1,
                                "verified": 0,
                                "falsified": 0,
                                "unknown": 1,
                                "coverage": 0,
                            }
                        },
                    },
                    "cost": {
                        "overall": {
                            "known_usage_calls": 2,
                            "unknown_usage_calls": 0,
                            "tokens": {"input": 10, "output": 8, "reasoning": 0},
                            "unknown_token_calls": {"input": 0, "output": 0, "reasoning": 0},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    output = run_dir / "metrics" / "ms2.json"

    result = write_ms2_analysis(trace_path, output, bootstrap_replicates=0)
    report = json.loads(result.read_text(encoding="utf-8"))

    assert output.with_name("ms2-coverage.png").is_file()
    assert report["evidence_coverage"]["coverage_status_counts"] == {"unverified": 3}
    assert report["evidence_coverage"]["claims_by_source"]["Unknown"]["unknown"] == 3
    assert report["costs"]["scoring_artifacts"] == 3
    assert report["costs"]["rollout_slots_with_usage"] == 6
    assert report["costs"]["usage_status"] == "known"
