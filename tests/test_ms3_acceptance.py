"""Acceptance-gate tests for the frozen MS3 Pilot contract."""

from __future__ import annotations

import json
from pathlib import Path

from ricl_experiment.analysis.ms3_pilot import write_ms3_pilot_artifacts
from ricl_experiment.analysis.ms3_repair import analyze_repairs
from ricl_experiment.cli import main
from ricl_experiment.config import fingerprint
from ricl_experiment.core.ledger import Ledger
from ricl_experiment.motivation.ms3.analysis import analyze_diagnoses
from ricl_experiment.motivation.ms3.protocol import METHODS, build_diagnosis
from ricl_experiment.motivation.ms3_repair_protocol import GROUPS, build_repair_request


def test_ms3_pilot_gate_writes_engineering_status_and_planning_artifacts(tmp_path: Path):
    manifest, audit, answer_key = _frozen_inputs()
    _write_frozen_inputs(tmp_path, manifest, audit, answer_key)
    diagnosis_dir, diagnosis_analysis = _write_diagnosis_artifacts(tmp_path, manifest, audit)
    repair_dir, repair_analysis = _write_repair_artifacts(tmp_path, manifest)
    output = tmp_path / "acceptance" / "status.json"
    plan_output = tmp_path / "acceptance" / "planning.json"

    paths = write_ms3_pilot_artifacts(
        manifest,
        audit,
        answer_key,
        diagnosis_dir,
        diagnosis_analysis,
        repair_dir,
        repair_analysis,
        output,
        plan_output=plan_output,
        pricing={"input_per_1k_tokens": 0.01, "output_per_1k_tokens": 0.02},
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    plan = json.loads(plan_output.read_text(encoding="utf-8"))
    assert paths == {"status": output, "plan": plan_output}
    assert report["status"] == "engineering_gate_passed"
    assert report["checks"]["sample_shape"]["observed"] == {
        "trajectory_count": 10,
        "question_group_count": 10,
        "error_trajectory_count": 8,
        "no_error_trajectory_count": 2,
    }
    assert report["checks"]["diagnosis_methods"]["status"] == "passed"
    assert all(
        details["legal_output_count"] == 10
        for details in report["checks"]["diagnosis_methods"]["methods"].values()
    )
    assert report["checks"]["repair_coverage"]["four_group_evaluable_count"] == 6
    assert report["checks"]["leakage"]["status"] == "passed"
    assert report["checks"]["ledger"]["status"] == "passed"
    assert report["checks"]["restart"]["status"] == "passed"
    assert report["checks"]["restart"]["resume_execution"] == "not_run"
    assert report["checks"]["offline_recompute"]["status"] == "passed"
    assert report["external_evidence"]["real_api"]["status"] == "not_provided"
    assert report["external_evidence"]["human_signoff"]["status"] == "not_provided"
    assert report["confirmatory_readiness"] == "not_ready_without_external_evidence"
    assert plan["cost"]["status"] == "estimated_from_recorded_tokens"
    assert plan["sample_size"]["comparisons"]["G3-G1"]["observed_evaluable_pairs"] == 8

    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(json.dumps({"input_per_1k_tokens": 0.01, "output_per_1k_tokens": 0.02}), encoding="utf-8")
    cli_output = tmp_path / "acceptance-cli" / "status.json"
    assert main([
        "ms3-pilot-check",
        "--manifest", str(tmp_path / "manifest.json"),
        "--audit", str(tmp_path / "audit.json"),
        "--answer-key", str(tmp_path / "answer-key.json"),
        "--diagnosis-run", str(diagnosis_dir),
        "--diagnosis-analysis", str(diagnosis_analysis),
        "--repair-run", str(repair_dir),
        "--repair-analysis", str(repair_analysis),
        "--output", str(cli_output),
        "--pricing", str(pricing_path),
    ]) == 0


def test_ms3_pilot_gate_reports_missing_external_record_explicitly(tmp_path: Path):
    manifest, audit, answer_key = _frozen_inputs()
    diagnosis_dir, diagnosis_analysis = _write_diagnosis_artifacts(tmp_path, manifest, audit)
    repair_dir, repair_analysis = _write_repair_artifacts(tmp_path, manifest)
    output = tmp_path / "status.json"

    write_ms3_pilot_artifacts(
        manifest,
        audit,
        answer_key,
        diagnosis_dir,
        diagnosis_analysis,
        repair_dir,
        repair_analysis,
        output,
        real_api_record=tmp_path / "missing-real-api.json",
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["external_evidence"]["real_api"]["status"] == "missing_record"
    assert report["external_evidence"]["real_api"]["credential_status"] == "not_provided"
    assert report["confirmatory_readiness"] == "not_ready_without_external_evidence"


def test_ms3_pilot_gate_reports_restricted_fields_in_model_requests(tmp_path: Path):
    manifest, audit, answer_key = _frozen_inputs()
    diagnosis_dir, diagnosis_analysis = _write_diagnosis_artifacts(tmp_path, manifest, audit)
    repair_dir, repair_analysis = _write_repair_artifacts(tmp_path, manifest)
    request_path = repair_dir / "restricted" / "requests.json"
    requests = json.loads(request_path.read_text(encoding="utf-8"))
    requests[0]["request"]["expert_label"] = 0
    request_path.write_text(json.dumps(requests), encoding="utf-8")
    output = tmp_path / "status.json"

    write_ms3_pilot_artifacts(
        manifest,
        audit,
        answer_key,
        diagnosis_dir,
        diagnosis_analysis,
        repair_dir,
        repair_analysis,
        output,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "engineering_gate_failed"
    assert report["checks"]["leakage"]["status"] == "failed"
    assert "expert_label" in report["checks"]["leakage"]["violations"][0]


def _frozen_inputs() -> tuple[dict, dict, list[dict]]:
    samples = [
        {
            "sample_id": f"sample-{index}",
            "trajectory_id": f"trajectory-{index}",
            "question_group_id": f"question-{index}",
            "problem": f"Problem {index}",
            "steps": ["first step", "second step", "third step"],
            "source": "processbench",
        }
        for index in range(10)
    ]
    labels = [0] * 8 + [-1] * 2
    manifest = {
        "schema_version": 1,
        "study": "ms3",
        "phase": "pilot",
        "manifest_id": "ms3-pilot-fixture",
        "fingerprint": "manifest-fixture",
        "samples": samples,
    }
    audit = {
        "manifest_fingerprint": manifest["fingerprint"],
        "processbench_labels": [
            {"trajectory_id": sample["trajectory_id"], "label": label}
            for sample, label in zip(samples, labels, strict=True)
        ],
        "study_pools": {
            "diagnosis": [sample["sample_id"] for sample in samples],
            "repair": [sample["sample_id"] for sample in samples[:8]],
        },
    }
    answer_key = [
        {
            "sample_id": sample["sample_id"],
            "expert_label": label,
            "final_answer_correct": label == -1,
            "reference_material": {
                "kind": "answer",
                "content": f"Private reference material {index}",
                "source": "fixture",
                "independently_verified": True,
            },
            "steps": sample["steps"],
        }
        for index, (sample, label) in enumerate(zip(samples, labels, strict=True))
    ]
    return manifest, audit, answer_key


def _write_frozen_inputs(tmp_path: Path, manifest: dict, audit: dict, answer_key: list[dict]) -> None:
    """Persist the fixture inputs in the same split shape accepted by the CLI."""
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (tmp_path / "answer-key.json").write_text(json.dumps(answer_key), encoding="utf-8")


def _write_diagnosis_artifacts(tmp_path: Path, manifest: dict, audit: dict) -> tuple[Path, Path]:
    run_dir = tmp_path / "diagnosis-run"
    (run_dir / "requests").mkdir(parents=True)
    (run_dir / "responses").mkdir()
    methods: dict[str, list[dict]] = {method["id"]: [] for method in METHODS}
    ledger = Ledger(run_dir / "run.sqlite3", 4096)
    try:
        for sample_index, sample in enumerate(manifest["samples"]):
            has_error = sample_index < 8
            for method in METHODS:
                request = build_diagnosis(
                    method,
                    sample,
                    "fixture-model",
                    {"max_tokens": 64, "temperature": 0},
                )
                call_id = f"diagnosis-{sample['sample_id']}-{method['id']}"
                request_fingerprint = fingerprint(request)
                request_ref = f"requests/{call_id}.json"
                (run_dir / request_ref).write_text(json.dumps(request), encoding="utf-8")
                ledger.ensure_call(call_id, sample["sample_id"], request_fingerprint)
                attempt_id = ledger.start_attempt(
                    call_id,
                    request_fingerprint,
                    {"max_tokens": 64, "temperature": 0},
                    64,
                    1,
                )
                diagnosis = {
                    "has_error": has_error,
                    "first_error_step": 0 if has_error else None,
                    "confidence": 1,
                }
                response_ref = f"responses/{attempt_id}.json"
                raw_response = {
                    "choices": [{"message": {"content": json.dumps(diagnosis)}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
                (run_dir / response_ref).write_text(json.dumps(raw_response), encoding="utf-8")
                ledger.record_response(call_id, attempt_id, response_ref, (10, 5))
                methods[method["id"]].append(
                    {
                        "sample_id": sample["sample_id"],
                        "trajectory_id": sample["trajectory_id"],
                        "question_group_id": sample["question_group_id"],
                        "method": method["id"],
                        "status": "completed",
                        "diagnosis": diagnosis,
                        "raw_output": json.dumps(diagnosis),
                        "usage": {"input_tokens": 10, "output_tokens": 5, "status": "known"},
                        "request": {
                            "logical_call_id": call_id,
                            "request_fingerprint": request_fingerprint,
                            "request_ref": request_ref,
                        },
                        "response_ref": response_ref,
                        "attempts": 1,
                    }
                )
        ledger.close_run("completed")
    finally:
        ledger.close()
    run = {
        "schema_version": 1,
        "run_id": "diagnosis-fixture",
        "study": "ms3",
        "manifest_id": manifest["manifest_id"],
        "manifest_fingerprint": manifest["fingerprint"],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "trajectory_id": sample["trajectory_id"],
                "question_group_id": sample["question_group_id"],
                "step_count": len(sample["steps"]),
            }
            for sample in manifest["samples"]
        ],
        "methods": methods,
        "status": "completed",
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    analysis = analyze_diagnoses(run, audit)
    analysis_path = run_dir / "metrics.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    return run_dir, analysis_path


def _write_repair_artifacts(tmp_path: Path, manifest: dict) -> tuple[Path, Path]:
    run_dir = tmp_path / "repair-run"
    (run_dir / "restricted").mkdir(parents=True)
    (run_dir / "blind").mkdir()
    (run_dir / "responses").mkdir()
    assignments = []
    evaluations = []
    request_audit = []
    ledger = Ledger(run_dir / "run.sqlite3", 4096)
    try:
        for sample_index, sample in enumerate(manifest["samples"][:8]):
            for group in GROUPS:
                location = None if group in {"G0", "G1"} else 1 if group == "G2" else 0
                request = build_repair_request(
                    sample["problem"], sample["steps"], group, location,
                    {"max_tokens": 64, "temperature": 0}, "fixture-model",
                )
                call_id = f"repair-{sample['sample_id']}-{group}"
                request_fingerprint = fingerprint(request)
                ledger.ensure_call(call_id, sample["sample_id"], request_fingerprint)
                attempt_id = ledger.start_attempt(
                    call_id,
                    request_fingerprint,
                    {"max_tokens": 64, "temperature": 0},
                    64,
                    1,
                )
                response_ref = f"responses/{attempt_id}.json"
                response = {"choices": [{"message": {"content": "repaired solution"}}]}
                (run_dir / response_ref).write_text(json.dumps(response), encoding="utf-8")
                ledger.record_response(call_id, attempt_id, response_ref, (10, 5))
                evaluation_id = f"evaluation-{sample_index}-{group}"
                assignments.append(
                    {
                        "evaluation_id": evaluation_id,
                        "sample_id": sample["sample_id"],
                        "trajectory_id": sample["trajectory_id"],
                        "question_group_id": sample["question_group_id"],
                        "group": group,
                        "call_id": call_id,
                        "g2_candidate_indices": [1, 2],
                        "g2_position": 1,
                        "first_error_position": 0,
                        "g3_position": 0,
                        "position_offset": 1,
                        "trajectory_length": 3,
                        "stratum": "fixture",
                    }
                )
                evaluations.append(
                    {
                        "evaluation_id": evaluation_id,
                        "repair_success": "unknown" if sample_index >= 6 and group == "G0" else "correct",
                        "error_removal": "removed",
                        "new_error": "absent",
                        "evidence": {"source": "fixture"},
                    }
                )
                request_audit.append({"call_id": call_id, "group": group, "request": request})
        ledger.close_run("completed")
    finally:
        ledger.close()
    (run_dir / "metadata.json").write_text(
        json.dumps({"status": "completed", "groups": list(GROUPS)}), encoding="utf-8"
    )
    (run_dir / "restricted" / "assignments.json").write_text(
        json.dumps(assignments), encoding="utf-8"
    )
    (run_dir / "restricted" / "requests.json").write_text(
        json.dumps(request_audit), encoding="utf-8"
    )
    (run_dir / "blind" / "evaluations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in evaluations), encoding="utf-8"
    )
    analysis = analyze_repairs(evaluations, assignments, bootstrap_samples=20, seed=1)
    analysis_path = run_dir / "metrics.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    return run_dir, analysis_path
