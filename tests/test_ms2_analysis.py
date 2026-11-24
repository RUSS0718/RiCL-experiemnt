from __future__ import annotations

import pytest

from ricl_experiment.analysis.ms2 import analyze_ms2_trace


def _candidate(candidate_id: str, y: tuple[str, str], rater: float, progress: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "route_id": candidate_id,
        "prefix_status": "available",
        "initial_evaluation": {"analysis_category": "early_incorrect"},
        "rollouts": [
            {"status": "completed", "evaluation": {"status": value}}
            for value in y
        ],
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
    assert report["comparisons"]["ranking"]["rater_plus_evidence_minus_rater"]["estimate"] == pytest.approx(1 / 6)
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

