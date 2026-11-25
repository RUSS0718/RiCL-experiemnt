"""CLI-level MS2 Checkpoint and independent continuation contracts."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from ricl_experiment.cli import main
from ricl_experiment.config import fingerprint


class MS2ControlledEndpoint(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    request_lock = threading.Lock()
    generation_count = 0
    continuation_count = 0
    empty_candidate_index: int | None = None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][0]["content"]
        with self.request_lock:
            is_continuation = "Checkpoint:" in content
            handler_type = type(self)
            if "Rater instructions:" in content:
                request_index, finish_reason = -1, "stop"
                response_text = '{"rating": 2, "rationale": "Uses the relevant equation."}'
            elif "Evidence instructions:" in content:
                request_index, finish_reason = -1, "stop"
                response_text = '{"claims": [{"assertion": "The equation can be evaluated."}]}'
            elif "Independent evidence review:" in content:
                request_index, finish_reason = -1, "stop"
                response_text = (
                    '{"reviews":[{"claim_id":"extracted-0","verdict":"verified",'
                    '"source_locator":"Step one","conditions_sufficient":true,'
                    '"rationale":"The operations are defined on integers.","milestone":2}]}'
                )
            elif is_continuation:
                request_index = handler_type.continuation_count
                handler_type.continuation_count += 1
                response_text = f"Continuation {request_index}\nFinal answer: 42"
                finish_reason = "stop"
            else:
                request_index = handler_type.generation_count
                handler_type.generation_count += 1
                if request_index == self.empty_candidate_index:
                    response_text = ""
                    finish_reason = "length"
                elif "stop before the final answer" in content:
                    response_text = "The polynomial separates into an even part and an odd term."
                    finish_reason = "stop"
                elif request_index == 0:
                    response_text = "Step one.\nFinal answer: 42"
                    finish_reason = "stop"
                elif request_index == 1:
                    response_text = "Step one.\nFinal answer: 41"
                    finish_reason = "stop"
                else:
                    response_text = "Step one, still checking the result."
                    finish_reason = "length"
            self.requests.append(
                {
                    "body": body,
                    "is_continuation": is_continuation,
                    "request_index": request_index,
                }
            )

        payload = {
            "id": f"controlled-{len(self.requests)}",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


@pytest.fixture
def ms2_endpoint():
    MS2ControlledEndpoint.requests = []
    MS2ControlledEndpoint.generation_count = 0
    MS2ControlledEndpoint.continuation_count = 0
    MS2ControlledEndpoint.empty_candidate_index = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), MS2ControlledEndpoint)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def write_ms2_inputs(
    tmp_path: Path, endpoint: str, *, budget: int = 72, phase: str = "pilot"
) -> tuple[Path, Path, Path, Path]:
    config = tmp_path / "model.toml"
    config.write_text(
        "\n".join(
            [
                "[model]",
                f'base_url = "{endpoint}"',
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
                "initial_max_tokens = 8",
                "continuation_max_tokens = 8",
                "",
                "[budget]",
                "max_attempts_per_call = 2",
                f"max_reserved_output_tokens = {budget}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "ms2-controlled",
                "study": "ms2",
                "phase": phase,
                "samples": [
                    {
                        "sample_id": "sample-1",
                        "problem": "What is 6 times 7?",
                        "question_group_id": "group-1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    answer_key = tmp_path / "answer-key.json"
    answer_key.write_text(
        json.dumps([{"sample_id": "sample-1", "reference_answer": "42"}]),
        encoding="utf-8",
    )
    return config, manifest, answer_key, tmp_path / "artifacts"


def run_ms2(
    command: str,
    config: Path,
    manifest: Path,
    answer_key: Path,
    artifacts: Path,
    run_id: str | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, "-m", "ricl_experiment", "ms2", command]
    if command == "run":
        phase = json.loads(manifest.read_text(encoding="utf-8"))["phase"]
        args += ["--phase", phase]
    else:
        args += ["--run-id", run_id or ""]
    args += [
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--answer-key",
        str(answer_key),
        "--artifacts-root",
        str(artifacts),
    ]
    return subprocess.run(args, capture_output=True, text=True, check=False)


def test_ms2_cli_commits_blinded_scoring_before_continuations(tmp_path, ms2_endpoint):
    config, manifest, answer_key, artifacts = write_ms2_inputs(tmp_path, ms2_endpoint, budget=144)
    config.write_text(
        config.read_text(encoding="utf-8").replace("[ms2]", "[ms2]\nscore_checkpoints = true"),
        encoding="utf-8",
    )
    result = run_ms2("run", config, manifest, answer_key, artifacts)
    assert result.returncode == 0, result.stderr
    run_id = result.stdout.strip().removeprefix("run_id=")
    requests = [row["body"]["messages"][0]["content"] for row in MS2ControlledEndpoint.requests]
    assert len(requests) == 18
    assert "Rater instructions:" in requests[1]
    assert "Evidence instructions:" in requests[2]
    assert "Independent evidence review:" in requests[3]
    assert '"rating"' not in requests[3]
    assert "Instruction:" in requests[4]
    run_dir = artifacts / "motivation" / run_id
    trace = json.loads((run_dir / "traces" / "ms2.json").read_text(encoding="utf-8"))
    candidate = trace["samples"][0]["candidates"][0]
    score = json.loads((run_dir / candidate["scoring_artifact_ref"]).read_text(encoding="utf-8"))
    assert score["verified_progress"]["score"] == 2
    assert score["verified_progress"]["claims"][0]["primary_provenance"] == "Agent-Assessed"
    assert score["associated_outcomes"] == []
    assert run_ms2("resume", config, manifest, answer_key, artifacts, run_id).returncode == 0
    assert len(MS2ControlledEndpoint.requests) == 18
    assert main(["ms2-pilot-check", "--run-dir", str(run_dir)]) == 2
    gate = json.loads((run_dir / "metrics" / "pilot-gate.json").read_text(encoding="utf-8"))
    assert gate["leakage_audit"]["status"] == "pass"
    assert gate["leakage_audit"]["request_count"] == 18
    assert gate["evidence_diagnostics"]["valid_scores"] == 3


def test_adjustment_validation_freezes_checkpoint_and_json_evidence_contract(tmp_path, ms2_endpoint):
    config, manifest, answer_key, artifacts = write_ms2_inputs(
        tmp_path, ms2_endpoint, budget=144, phase="adjustment_validation"
    )
    config.write_text(
        config.read_text(encoding="utf-8").replace("[ms2]", "[ms2]\nscore_checkpoints = true"),
        encoding="utf-8",
    )

    result = run_ms2("run", config, manifest, answer_key, artifacts)
    assert result.returncode == 0, result.stderr
    run_id = result.stdout.strip().removeprefix("run_id=")
    run_dir = artifacts / "motivation" / run_id
    trace = json.loads((run_dir / "traces" / "ms2.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    generation = [
        item["body"] for item in MS2ControlledEndpoint.requests
        if "stop before the final answer" in item["body"]["messages"][0]["content"]
    ]
    continuation = [
        item["body"] for item in MS2ControlledEndpoint.requests
        if item["body"]["messages"][0]["content"].startswith("Problem:")
        and "Instruction:\nContinue the visible work" in item["body"]["messages"][0]["content"]
    ]
    evidence = [
        item["body"] for item in MS2ControlledEndpoint.requests
        if "Evidence instructions:" in item["body"]["messages"][0]["content"]
    ]
    reviews = [
        item["body"] for item in MS2ControlledEndpoint.requests
        if "Independent evidence review:" in item["body"]["messages"][0]["content"]
    ]
    raters = [
        item["body"] for item in MS2ControlledEndpoint.requests
        if "Rater instructions:" in item["body"]["messages"][0]["content"]
    ]

    assert len(generation) == 3
    assert all(
        "Do not include a final-answer line" in body["messages"][0]["content"]
        for body in generation
    )
    assert "reference_answer" not in json.dumps(generation)
    assert len(continuation) == 6
    assert all(
        body["messages"][0]["content"].endswith(
            "Instruction:\nContinue the visible work, correct it if needed, and finish with exactly one final line in this format: Final answer: <answer>. Do not provide alternative or duplicate final answers."
        )
        for body in continuation
    )
    assert len(evidence) == 3
    assert len(reviews) == 3
    assert all('"claims"' in body["messages"][0]["content"] and "json" in body["messages"][0]["content"] for body in evidence)
    assert all('"reviews"' in body["messages"][0]["content"] and "json" in body["messages"][0]["content"] for body in reviews)
    assert len(raters) == 3
    assert all(
        body["response_format"] == {"type": "json_object"} for body in [*evidence, *reviews]
    )
    assert all("response_format" not in body for body in raters)
    assert all(
        not candidate["prefix_text"].lower().count("final answer")
        for candidate in trace["samples"][0]["candidates"]
    )
    assert trace["generation_template_version"] == "ms2-adjustment-checkpoint-v1"
    assert "stop before the final answer" in trace["generation_instructions"].lower()
    assert trace["continuation_template_version"] == "ms2-neutral-continuation-v2"
    assert metadata["generation_template_version"] == trace["generation_template_version"]
    assert metadata["prompt_fingerprint"] != fingerprint(["What is 6 times 7?"])
    assert metadata["scoring_template_versions"][1] == "ms2-evidence-extraction-json-v1"
    assert metadata["scoring_template_versions"][2] == "ms2-evidence-review-json-v1"

    score = json.loads(
        (run_dir / trace["samples"][0]["candidates"][0]["scoring_artifact_ref"]).read_text(
            encoding="utf-8"
        )
    )
    assert score["templates"]["evidence"] == "ms2-evidence-extraction-json-v1"
    assert score["templates"]["evidence_review"] == "ms2-evidence-review-json-v1"
    before_resume = len(MS2ControlledEndpoint.requests)
    assert run_ms2("resume", config, manifest, answer_key, artifacts, run_id).returncode == 0
    assert len(MS2ControlledEndpoint.requests) == before_resume


def test_ms2_cli_keeps_three_routes_and_two_isolated_rollouts(tmp_path, ms2_endpoint):
    config, manifest, answer_key, artifacts = write_ms2_inputs(tmp_path, ms2_endpoint)

    result = run_ms2("run", config, manifest, answer_key, artifacts)
    assert result.returncode == 0, result.stderr
    run_id = result.stdout.strip().removeprefix("run_id=")
    run_dir = artifacts / "motivation" / run_id
    trace = json.loads((run_dir / "traces" / "ms2.json").read_text(encoding="utf-8"))

    assert len(MS2ControlledEndpoint.requests) == 9
    generation_requests = [item for item in MS2ControlledEndpoint.requests if not item["is_continuation"]]
    continuation_requests = [item for item in MS2ControlledEndpoint.requests if item["is_continuation"]]
    assert len(generation_requests) == 3
    assert len(continuation_requests) == 6
    assert {item["body"]["messages"][0]["content"] for item in generation_requests} == {
        "What is 6 times 7?"
    }

    candidates = trace["samples"][0]["candidates"]
    assert len(candidates) == 3
    assert len({candidate["candidate_id"] for candidate in candidates}) == 3
    assert len({candidate["checkpoint_id"] for candidate in candidates}) == 3
    assert all(candidate["source_attempt_id"] for candidate in candidates)
    assert len({candidate["prefix_text"] for candidate in candidates}) == 3
    assert all(candidate["attempts"][0]["effective_parameters"]["max_tokens"] == 8 for candidate in candidates)
    assert [candidate["initial_evaluation"]["analysis_category"] for candidate in candidates] == [
        "early_correct",
        "early_incorrect",
        "early_unknown",
    ]

    for candidate in candidates:
        assert candidate["prefix_status"] == "available"
        assert candidate["prefix_fingerprint"]
        assert len(candidate["rollouts"]) == 2
        assert len({rollout["logical_call_id"] for rollout in candidate["rollouts"]}) == 2
        assert all(
            rollout["checkpoint_prefix_fingerprint"] == candidate["prefix_fingerprint"]
            for rollout in candidate["rollouts"]
        )
        assert all(
            rollout["evaluation_reference"]["answer_key_ref"] == "restricted/answer-key.json"
            for rollout in candidate["rollouts"]
        )

    for left, right in zip(continuation_requests[::2], continuation_requests[1::2]):
        assert left["body"] == right["body"]
        content = left["body"]["messages"][0]["content"]
        assert "reference_answer" not in content
        assert "expert_label" not in content
        assert "score" not in content
        assert "Continuation 0" not in content
        assert "Continuation 1" not in content
        assert content.endswith(
            "Instruction:\n"
            "Inspect the visible work, continue or correct it as needed, and explicitly state a final answer."
        )

    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM logical_calls").fetchone()[0] == 9
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 9
        assert db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0] == 9


def test_ms2_empty_checkpoint_is_recorded_without_continuation(tmp_path, ms2_endpoint):
    MS2ControlledEndpoint.empty_candidate_index = 1
    config, manifest, answer_key, artifacts = write_ms2_inputs(tmp_path, ms2_endpoint)

    result = run_ms2("run", config, manifest, answer_key, artifacts)
    assert result.returncode == 0, result.stderr
    run_id = result.stdout.strip().removeprefix("run_id=")
    run_dir = artifacts / "motivation" / run_id
    trace = json.loads((run_dir / "traces" / "ms2.json").read_text(encoding="utf-8"))
    empty = trace["samples"][0]["candidates"][1]

    assert len(MS2ControlledEndpoint.requests) == 7
    assert empty["prefix_text"] == ""
    assert empty["prefix_status"] == "empty"
    assert empty["truncation_status"] == "truncated"
    assert empty["stop_reason"] == "length"
    assert empty["attempts"][0]["status"] == "completed"
    assert [rollout["failure"] for rollout in empty["rollouts"]] == [
        "empty_prefix",
        "empty_prefix",
    ]
    assert all(rollout["status"] == "not_dispatched" for rollout in empty["rollouts"])
    assert [segment["status"] for segment in empty["segments"]] == [
        "completed",
        "not_dispatched",
        "not_dispatched",
    ]


def test_ms2_resume_reuses_completed_calls_and_trace(tmp_path, ms2_endpoint):
    config, manifest, answer_key, artifacts = write_ms2_inputs(tmp_path, ms2_endpoint)
    started = run_ms2("run", config, manifest, answer_key, artifacts)
    assert started.returncode == 0, started.stderr
    run_id = started.stdout.strip().removeprefix("run_id=")
    run_dir = artifacts / "motivation" / run_id
    trace_path = run_dir / "traces" / "ms2.json"
    before_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("generation_template_version")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    resumed = run_ms2("resume", config, manifest, answer_key, artifacts, run_id)
    assert resumed.returncode == 0, resumed.stderr
    assert len(MS2ControlledEndpoint.requests) == 9
    after_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert after_trace["samples"] == before_trace["samples"]



def test_adjustment_checkpoint_detects_explicit_numeric_result_in_prose():
    from ricl_experiment.evaluation.answer import classify_answer
    from ricl_experiment.motivation.ms2 import _prefix_for_answer_evaluation

    prefix = r"We stop here as instructed: the result is \(23\), without a final answer line."
    control = r"Here is a local result: the sum is \(23\)."
    assert classify_answer(_prefix_for_answer_evaluation(prefix, "adjustment_validation"), "23").status == "correct"
    assert _prefix_for_answer_evaluation(control, "adjustment_validation") == control
    assert _prefix_for_answer_evaluation(prefix, "pilot") == prefix
    conflicting = prefix + "\nFinal answer: 24"
    assert classify_answer(_prefix_for_answer_evaluation(conflicting, "adjustment_validation"), "23").status == "blind_review"
