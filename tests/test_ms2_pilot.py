"""Offline MS2 Pilot harness contracts using deterministic local fixtures."""

from __future__ import annotations

import copy
import json
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from ricl_experiment.motivation.ms2 import start_ms2_run
from ricl_experiment.motivation.ms2_pilot import (
    build_pilot_report,
    recompute_ms2_metrics,
)


class ControlledMS2Handler(BaseHTTPRequestHandler):
    generation_counts: ClassVar[defaultdict[str, int]] = defaultdict(int)
    rollout_counts: ClassVar[defaultdict[tuple[str, int], int]] = defaultdict(int)
    requests: ClassVar[list[dict]] = []
    lock = threading.Lock()

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][0]["content"]
        with type(self).lock:
            if "Checkpoint:" not in content:
                sample_id = content
                candidate_index = type(self).generation_counts[sample_id]
                type(self).generation_counts[sample_id] += 1
                text = f"Checkpoint: {sample_id} candidate {candidate_index}"
            else:
                sample_id = content.split("Problem:\n", 1)[1].split("\n\n", 1)[0]
                checkpoint = next(
                    line for line in content.splitlines() if line.startswith("Checkpoint: ")
                )
                candidate_index = int(checkpoint.rsplit(" ", 1)[1])
                key = (sample_id, candidate_index)
                rollout_index = type(self).rollout_counts[key]
                type(self).rollout_counts[key] += 1
                sample_index = int(sample_id.rsplit("-", 1)[1])
                patterns = ((True, False), (False, True), (True, True), (False, False))
                success = patterns[(sample_index + candidate_index) % len(patterns)][rollout_index]
                text = "Final answer: 1" if success else "Final answer: 0"
            type(self).requests.append(body)

        payload = {
            "id": f"controlled-{len(type(self).requests)}",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _manifest() -> dict:
    samples = []
    for index in range(10):
        source = "math500" if index < 5 else "omni_math"
        samples.append(
            {
                "sample_id": f"sample-{index}",
                "problem": f"Problem sample-{index}",
                "source": source,
                "revision": "fixture-revision",
                "split": "test",
                "source_record_id": f"{source}:record-{index}",
                "question_group_id": f"group-{index}",
            }
        )
    return {
        "schema_version": 1,
        "manifest_id": "ms2-pilot-fixture",
        "study": "ms2",
        "phase": "pilot",
        "samples": samples,
        "sampling": {"seed": 17},
    }


def _trace_and_scoring() -> tuple[dict, list[dict]]:
    manifest = _manifest()
    trace_samples = []
    scoring = []
    patterns = ((True, False), (False, True), (True, True), (False, False))
    for sample_index, sample in enumerate(manifest["samples"]):
        candidates = []
        for candidate_index in range(3):
            candidate_id = f"candidate-{sample_index}-{candidate_index}"
            outcomes = patterns[(sample_index + candidate_index) % len(patterns)]
            rollouts = [
                {
                    "rollout_index": rollout_index,
                    "status": "completed",
                    "checkpoint_prefix_fingerprint": f"prefix-{sample_index}-{candidate_index}",
                    "text": f"Final answer: {1 if success else 0}",
                    "evaluation": {"status": "correct" if success else "incorrect"},
                }
                for rollout_index, success in enumerate(outcomes)
            ]
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    "route_id": f"route-{sample_index}-{candidate_index}",
                    "problem_id": sample["sample_id"],
                    "checkpoint_id": f"checkpoint-{sample_index}-{candidate_index}",
                    "prefix_text": f"Checkpoint: {sample['sample_id']} candidate {candidate_index}",
                    "prefix_fingerprint": f"prefix-{sample_index}-{candidate_index}",
                    "prefix_status": "available",
                    "truncation_status": "not_truncated",
                    "last_step_status": "complete",
                    "initial_evaluation": {"status": "unknown"},
                    "rollouts": rollouts,
                    "segments": [],
                }
            )
            scoring.append(
                {
                    "sample_id": sample["sample_id"],
                    "candidate_id": candidate_id,
                    "rater_score": (candidate_index + sample_index) % 4,
                    "evidence_score": (3 - candidate_index + sample_index) % 4,
                    "evidence_source": "Tool-Verified" if sample_index < 5 and candidate_index == 0 else "Unknown",
                    "evidence_verdict": "confirmed" if sample_index < 5 and candidate_index == 0 else "unknown",
                    "independent_evidence": sample_index < 5 and candidate_index == 0,
                }
            )
        trace_samples.append({**sample, "candidates": candidates})
    trace = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "study": "ms2",
        "phase": "pilot",
        "manifest_id": manifest["manifest_id"],
        "manifest_fingerprint": "fixture-fingerprint",
        "candidate_slots": 3,
        "rollouts_per_checkpoint": 2,
        "continuation_template_version": "ms2-neutral-continuation-v1",
        "samples": trace_samples,
        "usage": {"known_output_tokens": 270, "reserved_output_tokens": 0, "unknown_usage_attempts": 0},
    }
    return trace, scoring


def _requests() -> list[dict]:
    return [
        {"messages": [{"role": "user", "content": f"Problem sample-{index}"}]}
        for index in range(10)
    ]


def _resume_evidence() -> dict:
    return {
        "logical_calls": [
            {
                "logical_call_id": "logical-recovered",
                "attempts": [
                    {"attempt_number": 1, "status": "unknown"},
                    {"attempt_number": 2, "status": "completed"},
                ],
                "analysis_results": [{"logical_call_id": "logical-recovered"}],
            }
        ]
    }


def _execution_limits() -> dict:
    return {"max_attempts_per_call": 1, "max_reserved_output_tokens": 720}


def _planning() -> dict:
    return {
        "targets": {
            "route_selection_success": {
                "minimum_meaningful_difference": 0.10,
                "target_half_width": 0.20,
            },
            "within_problem_ranking_accuracy": {
                "minimum_meaningful_difference": 0.10,
                "target_half_width": 0.20,
            },
        },
        "cost_rates": {
            "input_per_1k": 0.01,
            "output_per_1k": 0.02,
        },
        "usage": {"input_tokens": 630, "output_tokens": 270},
        "conservative": {"sd_multiplier": 1.5, "valid_rate_multiplier": 0.8},
        "max_budget": 100.0,
    }


def test_pilot_metrics_use_only_eligible_routes_and_read_live_scoring_artifacts() -> None:
    trace, scoring = _trace_and_scoring()
    first, second = trace["samples"][:2]
    first["candidates"][2]["initial_evaluation"] = {"status": "correct"}
    second["candidates"][0]["initial_evaluation"] = {"status": "correct"}
    second["candidates"][1]["initial_evaluation"] = {
        "status": "blind_review",
        "analysis_category": "early_unknown",
        "extracted_answer": "unresolved",
    }
    for candidate in first["candidates"][:2]:
        row = next(row for row in scoring if row.get("candidate_id") == candidate["candidate_id"])
        row.clear()
        row.update({
            "route_id": candidate["route_id"],
            "llm_rater": {"score": 1},
            "verified_progress": {"score": 1, "coverage_numerator": 1},
        })
    report = recompute_ms2_metrics(trace, scoring)
    first_topic = report["topics"][first["sample_id"]]
    second_topic = report["topics"][second["sample_id"]]
    assert first_topic["main_complete"]
    assert first_topic["independent_evidence"]
    assert len(first_topic["candidates"]) == 2
    assert not second_topic["main_complete"]


def test_pilot_report_reports_the_frozen_manifest_phase() -> None:
    manifest = _manifest()
    manifest["phase"] = "adjustment_validation"
    trace, scoring = _trace_and_scoring()
    trace["phase"] = "adjustment_validation"

    report = build_pilot_report(manifest, trace, scoring=scoring)

    assert report["phase"] == "adjustment_validation"
    assert report["shape"]["status"] == "pass"
    assert report["coverage"]["status"] == "pass"

    trace["phase"] = "pilot"
    mismatched = build_pilot_report(manifest, trace, scoring=scoring)
    assert "trace_identity" in mismatched["coverage"]["failures"]


def test_pilot_report_distinguishes_engineering_ready_from_external_blockers() -> None:
    manifest = _manifest()
    trace, scoring = _trace_and_scoring()
    metrics = recompute_ms2_metrics(trace, scoring)
    report = build_pilot_report(
        manifest,
        trace,
        requests=_requests(),
        resume_evidence=_resume_evidence(),
        scoring=scoring,
        reported_metrics=metrics,
        planning=_planning(),
        execution_limits=_execution_limits(),
    )

    assert report["status"] == "engineering_ready_externally_blocked"
    assert report["engineering_ready"] is True
    assert report["external_blocked"] is True
    assert set(report["external_blockers"]) == {
        "real_api_smoke_missing",
        "human_blind_review_missing",
        "budget_signoff_missing",
    }
    assert report["shape"]["status"] == "pass"
    assert report["shape"]["observed"] == {"total": 10, "math500": 5, "omni_math": 5}
    assert report["coverage"]["checkpoints"]["expected"] == 30
    assert report["coverage"]["rollouts"]["expected"] == 60
    assert report["thresholds"]["complete_topics"]["pass"] is True
    assert report["thresholds"]["ranking_topics"]["pass"] is True
    assert report["thresholds"]["independent_evidence_topics"]["pass"] is True
    assert report["leakage_audit"]["status"] == "pass"
    assert report["resume_evidence"]["status"] == "pass"
    assert report["offline_recomputation"]["status"] == "consistent"
    assert report["precision_planning"]["status"] == "computed"
    assert report["evidence_diagnostics"]["independent_evidence_topics"] == 5


def test_pilot_report_is_deterministic_and_mismatch_blocks_engineering() -> None:
    manifest = _manifest()
    trace, scoring = _trace_and_scoring()
    metrics = recompute_ms2_metrics(trace, scoring)
    expected = build_pilot_report(
        manifest,
        trace,
        requests=_requests(),
        resume_evidence=_resume_evidence(),
        scoring=scoring,
        reported_metrics=metrics,
        planning=_planning(),
        execution_limits=_execution_limits(),
    )
    changed = copy.deepcopy(metrics)
    changed["route_selection_success"]["rater"]["value"] += 0.01
    actual = build_pilot_report(
        manifest,
        trace,
        requests=_requests(),
        resume_evidence=_resume_evidence(),
        scoring=scoring,
        reported_metrics=changed,
        planning=_planning(),
    )

    assert actual["engineering_ready"] is False
    assert actual["offline_recomputation"]["status"] == "mismatch"
    assert json.dumps(expected, sort_keys=True) == json.dumps(
        build_pilot_report(
            manifest,
            trace,
            requests=_requests(),
            resume_evidence=_resume_evidence(),
            scoring=scoring,
            reported_metrics=metrics,
            planning=_planning(),
            execution_limits=_execution_limits(),
        ),
        sort_keys=True,
    )


def test_pilot_report_rejects_wrong_frozen_shape_and_missing_external_artifacts() -> None:
    manifest = _manifest()
    manifest["samples"][-1]["source"] = "math500"
    trace, scoring = _trace_and_scoring()
    report = build_pilot_report(manifest, trace, scoring=scoring)

    assert report["engineering_ready"] is False
    assert report["shape"]["status"] == "fail"
    assert "source_counts" in report["shape"]["failures"]
    assert report["leakage_audit"]["status"] == "missing"
    assert report["resume_evidence"]["status"] == "missing"
    assert report["offline_recomputation"]["status"] == "not_checked"
    assert report["external_blocked"] is True


def test_controlled_local_runner_trace_is_consumable_without_real_readiness_claim(
    tmp_path: Path,
) -> None:
    ControlledMS2Handler.generation_counts = defaultdict(int)
    ControlledMS2Handler.rollout_counts = defaultdict(int)
    ControlledMS2Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ControlledMS2Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manifest = _manifest()
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        answer_key_path = tmp_path / "answer-key.json"
        answer_key_path.write_text(
            json.dumps([{"sample_id": row["sample_id"], "reference_answer": 1} for row in manifest["samples"]]),
            encoding="utf-8",
        )
        config_path = tmp_path / "model.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[model]",
                    f'base_url = "http://127.0.0.1:{server.server_port}/v1"',
                    'model = "controlled-model"',
                    "",
                    "[model.capabilities]",
                    'supported_parameters = ["max_tokens", "temperature"]',
                    'required_parameters = ["max_tokens"]',
                    "",
                    "[model.parameters]",
                    "max_tokens = 8",
                    "temperature = 0",
                    "",
                    "[ms2]",
                    'protocol_version = "legacy-v1"',
                    "initial_max_tokens = 8",
                    "continuation_max_tokens = 8",
                    "",
                    "[budget]",
                    "max_attempts_per_call = 1",
                    "max_reserved_output_tokens = 720",
                ]
            ),
            encoding="utf-8",
        )
        artifacts_root = tmp_path / "artifacts"
        run_id, status = start_ms2_run(
            "pilot", config_path, manifest_path, answer_key_path, artifacts_root
        )
        run_dir = artifacts_root / "motivation" / run_id
        trace = json.loads((run_dir / "traces" / "ms2.json").read_text(encoding="utf-8"))
        scoring = []
        for sample_index, sample in enumerate(trace["samples"]):
            for candidate in sample["candidates"]:
                candidate_index = candidate["candidate_index"]
                scoring.append(
                    {
                        "sample_id": sample["sample_id"],
                        "candidate_id": candidate["candidate_id"],
                        "rater_score": (candidate_index + sample_index) % 4,
                        "evidence_score": (3 - candidate_index + sample_index) % 4,
                        "evidence_source": "Tool-Verified" if sample_index < 5 and candidate_index == 0 else "Unknown",
                        "evidence_verdict": "confirmed" if sample_index < 5 and candidate_index == 0 else "unknown",
                    }
                )
        report = build_pilot_report(
            manifest,
            trace,
            requests=ControlledMS2Handler.requests,
            scoring=scoring,
            resume_evidence=_resume_evidence(),
            reported_metrics=recompute_ms2_metrics(trace, scoring),
            planning=_planning(),
            execution_limits=_execution_limits(),
        )

        assert status == "completed"
        assert len(ControlledMS2Handler.requests) == 90
        assert report["coverage"]["rollouts"]["completed"] == 60
        assert report["engineering_ready"] is True
        assert report["external_blocked"] is True
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("artifact", [None, {}])
def test_external_evidence_is_never_invented(artifact) -> None:
    manifest = _manifest()
    trace, scoring = _trace_and_scoring()
    report = build_pilot_report(
        manifest,
        trace,
        requests=_requests(),
        resume_evidence=_resume_evidence(),
        scoring=scoring,
        reported_metrics=recompute_ms2_metrics(trace, scoring),
        planning=_planning(),
        external_evidence=artifact,
    )
    assert report["external_blocked"] is True
    assert report["external_evidence"]["real_api_smoke"]["status"] == "missing"
