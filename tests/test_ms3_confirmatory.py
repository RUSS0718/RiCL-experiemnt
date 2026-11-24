from __future__ import annotations

import pytest

from ricl_experiment.motivation.ms3_confirmatory import build_confirmatory_report

METHODS = ("direct_judge", "step_by_step_critic", "reflection")
GROUPS = ("G0", "G1", "G2", "G3")


def _diagnosis_run() -> dict:
    samples = []
    methods = {method: [] for method in METHODS}
    labels = []
    for index in range(2):
        sample_id = f"trace-{index}"
        samples.append({
            "sample_id": sample_id,
            "trajectory_id": sample_id,
            "question_group_id": f"question-{index}",
            "step_count": 3,
        })
        labels.append({
            "trajectory_id": sample_id,
            "question_group_id": f"question-{index}",
            "first_error_step": 1,
        })
        for method in METHODS:
            methods[method].append({
                "sample_id": sample_id,
                "status": "completed",
                "diagnosis": {"has_error": True, "first_error_step": 1},
            })
    return {
        "schema_version": 1,
        "study": "ms3",
        "phase": "confirmatory",
        "run_id": "confirm-1",
        "manifest_id": "manifest-1",
        "manifest_fingerprint": "digest-1",
        "samples": samples,
        "methods": methods,
    }, {"manifest_fingerprint": "digest-1", "processbench_labels": labels}


def _repair_rows() -> tuple[list[dict], list[dict]]:
    assignments = []
    evaluations = []
    for index in range(2):
        for group in GROUPS:
            evaluation_id = f"eval-{index}-{group}"
            assignments.append({
                "evaluation_id": evaluation_id,
                "sample_id": f"trace-{index}",
                "question_group_id": f"question-{index}",
                "group": group,
                "first_error_position": 1,
                "g2_position": 0,
                "g3_position": 1,
                "trajectory_length": 3,
                "stratum": "source-a",
            })
            evaluations.append({
                "evaluation_id": evaluation_id,
                "repair_success": "correct" if group == "G3" else "incorrect",
                "error_removal": "removed" if group == "G3" else "not_removed",
                "new_error": "absent",
            })
    return assignments, evaluations


def test_confirmatory_report_combines_metrics_and_completeness():
    diagnosis, audit = _diagnosis_run()
    assignments, evaluations = _repair_rows()
    report = build_confirmatory_report(
        diagnosis,
        audit,
        assignments,
        evaluations,
        expected_run_id="confirm-1",
        expected_manifest_fingerprint="digest-1",
        real_api_verified=False,
        human_review_verified=False,
    )

    assert report["status"] == "blocked_external_evidence"
    assert report["identity"]["same_run_id"] is True
    assert report["identity"]["same_manifest_fingerprint"] is True
    assert report["h3a"]["trajectory_count"] == 2
    assert report["h3b"]["four_group_common"]["n_samples"] == 2
    assert report["coverage"]["repair"]["complete_four_group_samples"] == 2
    assert report["strata"]["by_stratum"]["source-a"]["position_pairs"] == 2
    assert "real API" in " ".join(report["external_evidence"]["blocking_reasons"])


def test_confirmatory_report_rejects_mismatched_frozen_identity():
    diagnosis, audit = _diagnosis_run()
    assignments, evaluations = _repair_rows()
    with pytest.raises(ValueError, match="manifest fingerprint"):
        build_confirmatory_report(
            diagnosis,
            audit,
            assignments,
            evaluations,
            expected_run_id="confirm-1",
            expected_manifest_fingerprint="other-digest",
        )


def test_confirmatory_report_counts_missing_groups_and_diagnosis_failures():
    diagnosis, audit = _diagnosis_run()
    diagnosis["methods"]["reflection"][1]["status"] = "api_failure"
    diagnosis["methods"]["reflection"][1]["diagnosis"] = None
    assignments, evaluations = _repair_rows()
    assignments = [row for row in assignments if not (row["sample_id"] == "trace-1" and row["group"] == "G2")]
    report = build_confirmatory_report(
        diagnosis,
        audit,
        assignments,
        evaluations,
        expected_run_id="confirm-1",
        expected_manifest_fingerprint="digest-1",
    )

    assert report["coverage"]["diagnosis"]["method_status_counts"]["reflection"]["api_failure"] == 1
    assert report["coverage"]["repair"]["missing_group_assignments"] == 1
    assert report["h3b"]["four_group_common"]["n_samples"] == 1
