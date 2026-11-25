"""Contract tests for blinded MS3 diagnosis and offline metrics."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from ricl_experiment.config import RunConfig
from ricl_experiment.motivation.ms3.analysis import analyze_diagnoses
from ricl_experiment.motivation.ms3.diagnosis import (
    METHODS,
    build_diagnosis,
    parse_diagnosis,
    run_diagnosis,
)


class DiagnosisEndpoint(BaseHTTPRequestHandler):
    """Capture local test requests and return a deterministic structured diagnosis."""

    requests: ClassVar[list[dict]] = []
    status_code = 200
    lock = threading.Lock()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            self.requests.append(body)
        content = json.dumps(
            {
                "has_error": True,
                "first_error_step": 0,
                "error_type": None,
                "explanation": None,
                "confidence": 1,
            }
        )
        payload = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        """Keep test server output out of pytest logs."""


@pytest.fixture
def diagnosis_endpoint():
    """Provide a local-only endpoint so tests cannot incur model charges."""
    DiagnosisEndpoint.requests = []
    DiagnosisEndpoint.status_code = 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), DiagnosisEndpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _diagnosis_reply():
    from ricl_experiment.motivation.ms3.protocol import DiagnosisReply

    content = json.dumps({"has_error": True, "first_error_step": 0})
    raw = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    return DiagnosisReply(content, raw, {"input_tokens": 11, "output_tokens": 7}, 1)


def test_parse_diagnosis_accepts_all_four_contract_states():
    """Accept alarm, no-error, and abstention structures with legal positions."""
    assert parse_diagnosis('{"has_error":true,"first_error_step":0}', 2) == (
        "completed",
        {"has_error": True, "first_error_step": 0},
        None,
    )
    assert parse_diagnosis('{"has_error":true,"first_error_step":null}', 2)[0] == "completed"
    assert parse_diagnosis('{"has_error":false,"first_error_step":null}', 2)[0] == "completed"
    assert parse_diagnosis('{"has_error":null,"first_error_step":null}', 2)[0] == "abstained"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"has_error":true,"first_error_step":2,"error_type":null,"explanation":null,"confidence":1}',
        '{"has_error":false,"first_error_step":0,"error_type":null,"explanation":null,"confidence":1}',
        '{"has_error":null,"first_error_step":1,"error_type":null,"explanation":null,"confidence":null}',
        '{"has_error":1,"first_error_step":null,"error_type":null,"explanation":null,"confidence":null}',
        '{"has_error":true,"first_error_step":false,"error_type":null,"explanation":null,"confidence":1}',
        '{"has_error":true,"first_error_step":null,"error_type":null,"explanation":null,"confidence":1,"unexpected":1}',
        '{"has_error":true,"first_error_step":null,"error_type":null,"explanation":null,"confidence":null}',
    ],
)
def test_parse_diagnosis_distinguishes_parse_and_schema_failures(raw: str):
    """Reject malformed JSON separately from invalid structured output."""
    status, diagnosis, error = parse_diagnosis(raw, 2)
    if raw == "not json":
        assert status == "parse_error"
    else:
        assert status == "invalid_output"
    assert diagnosis is None
    assert error


def test_diagnosis_request_preserves_steps_and_excludes_private_fields():
    """Serialize the exact numbered trace without labels, answers, or tool access."""
    body = build_diagnosis(
        METHODS[0],
        {
            "sample_id": "trace-1",
            "problem": "Solve x.",
            "steps": ["  first\nline  ", "second"],
        },
        "test-model",
        {"max_tokens": 64, "temperature": 0},
    )
    content = body["messages"][0]["content"]
    assert "Step 0:\n  first\nline  " in content
    assert "Step 1:\nsecond" in content
    assert "expert_label" not in content
    assert "reference_answer" not in content
    assert "trace-1" not in content
    assert "tool_choice" not in body or body["tool_choice"] == "none"
    assert body["tool_choice"] == "none"
    assert body["model"] == "test-model"
    assert body["max_tokens"] == 64


def test_run_diagnosis_pairs_three_blind_requests_and_saves_audit_artifacts(
    tmp_path: Path, diagnosis_endpoint: str
):
    """Dispatch one isolated call per method and preserve request/response/use records."""
    config = RunConfig.model_validate(
        {
            "model": {
                "base_url": diagnosis_endpoint,
                "model": "local-test-model",
                "capabilities": {
                    "supported_parameters": ["max_tokens", "temperature"],
                    "required_parameters": ["max_tokens"],
                },
                "parameters": {"max_tokens": 64, "temperature": 0},
            },
            "budget": {"max_attempts_per_call": 1, "max_reserved_output_tokens": 512},
        }
    )
    manifest = {
        "study": "ms3",
        "manifest_id": "pilot-fixture",
        "fingerprint": "fixture-digest",
        "samples": [
            {
                "sample_id": "sample-a",
                "source": "processbench",
                "trajectory_id": "trace-a",
                "question_group_id": "question-a",
                "problem": "Solve 2 + 2.",
                "steps": ["  Add 2 and 2.  ", "Return 4."],
                "label": 0,
                "final_answer_correct": False,
                "reference_answer": "4",
            }
        ],
    }
    output = tmp_path / "diagnosis-run"

    run = run_diagnosis(config, manifest, output)

    assert len(DiagnosisEndpoint.requests) == 3
    assert set(run["methods"]) == {method["id"] for method in METHODS}
    for request in DiagnosisEndpoint.requests:
        content = request["messages"][0]["content"]
        assert "Solve 2 + 2." in content
        assert "Step 0:\n  Add 2 and 2.  " in content
        assert "reference_answer" not in content
        assert "final_answer_correct" not in content
        assert '"label": 0' not in content
        assert request["tool_choice"] == "none"
        assert request["model"] == "local-test-model"
        assert request["max_tokens"] == 64
    assert len({request["messages"][0]["content"].split("\n", 1)[0] for request in DiagnosisEndpoint.requests}) == 3
    for method in METHODS:
        row = run["methods"][method["id"]][0]
        assert row["status"] == "completed"
        assert row["diagnosis"] == {
            "has_error": True,
            "first_error_step": 0,
            "error_type": None,
            "explanation": None,
            "confidence": 1.0,
        }
        assert row["usage"] == {"input_tokens": 11, "output_tokens": 7, "status": "known"}
        assert (output / row["request"]["request_ref"]).is_file()
        assert (output / row["response_ref"]).is_file()
    saved = json.loads((output / "run.json").read_text(encoding="utf-8"))
    saved_json = json.dumps(saved)
    assert "expert_label" not in saved_json
    assert "final_answer_correct" not in saved_json
    assert "reference_answer" not in saved_json
    assert "Solve 2 + 2." not in saved_json
    assert "Add 2 and 2." not in saved_json


def test_diagnosis_resume_reuses_saved_calls_and_recovers_interrupted_attempt(tmp_path):
    """A resumed diagnosis run reuses durable responses and retries only an interrupted call."""
    config = RunConfig.model_validate(
        {
            "model": {
                "base_url": "http://localhost/v1",
                "model": "offline-test",
                "capabilities": {"supported_parameters": ["max_tokens"], "required_parameters": ["max_tokens"]},
                "parameters": {"max_tokens": 64},
            },
            "budget": {"max_attempts_per_call": 2, "max_reserved_output_tokens": 1024},
        }
    )
    manifest = {
        "study": "ms3",
        "manifest_id": "resume-fixture",
        "fingerprint": "resume-digest",
        "samples": [
            {
                "sample_id": "sample-a",
                "source": "processbench",
                "trajectory_id": "trace-a",
                "question_group_id": "question-a",
                "problem": "Solve 2 + 2.",
                "steps": ["Add 2 and 2.", "Return 4."],
            }
        ],
    }
    output = tmp_path / "diagnosis-resume"

    class InterruptedTransport:
        def __init__(self):
            self.calls = 0

        def send(self, request_body):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt("simulated process interruption")
            return _diagnosis_reply()

    first_transport = InterruptedTransport()
    with pytest.raises(KeyboardInterrupt):
        run_diagnosis(config, manifest, output, transport=first_transport)

    class CountingTransport:
        def __init__(self):
            self.calls = 0

        def send(self, request_body):
            self.calls += 1
            return _diagnosis_reply()

    resumed_transport = CountingTransport()
    run = run_diagnosis(config, manifest, output, transport=resumed_transport, resume=True)

    assert first_transport.calls == 2
    assert resumed_transport.calls == 2
    assert run["status"] == "completed"
    assert all(len(rows) == 1 for rows in run["methods"].values())
    with sqlite3.connect(output / "run.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM logical_calls").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0] == 3


def test_offline_analysis_matches_fixed_denominator_acceptance_example():
    """Recompute the agreed four-trace metrics, including conditional distances."""
    labels = {"correct": -1, "first": 0, "late": 1, "unparsed": 2}
    samples = [
        {
            "sample_id": sample_id,
            "trajectory_id": sample_id,
            "question_group_id": sample_id,
            "step_count": 4,
        }
        for sample_id in labels
    ]
    outcomes = {
        "correct": ("completed", {"has_error": False, "first_error_step": None}),
        "first": ("completed", {"has_error": True, "first_error_step": 0}),
        "late": ("completed", {"has_error": True, "first_error_step": 3}),
        "unparsed": ("parse_error", None),
    }
    methods = {
        method["id"]: [
            {
                "sample_id": sample["sample_id"],
                "status": outcomes[sample["sample_id"]][0],
                "diagnosis": outcomes[sample["sample_id"]][1],
            }
            for sample in samples
        ]
        for method in METHODS
    }
    audit = {
        "processbench_labels": [
            {"trajectory_id": trajectory_id, "label": label}
            for trajectory_id, label in labels.items()
        ]
    }

    analysis = analyze_diagnoses({"samples": samples, "methods": methods}, audit)

    for method in METHODS:
        metrics = analysis["methods"][method["id"]]
        assert metrics["confusion"] == {"tp": 2, "fp": 0, "fn": 1, "tn": 1}
        assert metrics["detection_f1"]["value"] == pytest.approx(0.8)
        assert metrics["first_error_accuracy"]["value"] == pytest.approx(1 / 3)
        assert metrics["false_positive_rate"]["value"] == 0
        assert metrics["localization_coverage"]["value"] == pytest.approx(2 / 3)
        assert metrics["late_report_rate"]["value"] == pytest.approx(1 / 3)
        assert metrics["localization_distance"]["signed"] == [0, 2]
        assert metrics["localization_distance"]["absolute"] == [0, 2]
        assert metrics["status_counts"]["parse_error"] == 1
    assert analysis["trajectory_count"] == 4
    assert analysis["question_group_count"] == 4
    assert analysis["paired_traces"][2]["question_group_id"] == "late"
    assert analysis["uncertainty"]["paired_differences"]
    assert analysis["uncertainty"]["cluster"] == "question_group_id"


def test_true_alarm_without_location_counts_for_detection_not_localization():
    """Count true/null as a detection TP while retaining zero localization coverage."""
    sample = {
        "sample_id": "s",
        "trajectory_id": "t",
        "question_group_id": "q",
        "step_count": 2,
    }
    run = {
        "samples": [sample],
        "methods": {
            method["id"]: [
                {
                    "sample_id": "s",
                    "status": "completed",
                    "diagnosis": {"has_error": True, "first_error_step": None},
                }
            ]
            for method in METHODS
        },
    }
    result = analyze_diagnoses(run, {"processbench_labels": [{"trajectory_id": "t", "label": 1}]})
    metric = result["methods"][METHODS[0]["id"]]
    assert metric["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}
    assert metric["localization_coverage"]["value"] == 0
    assert metric["first_error_accuracy"]["value"] == 0


def test_analysis_ignores_unselected_restricted_audit_rows():
    """Use only selected trace labels when the audit also contains excluded records."""
    sample = {"sample_id": "s", "trajectory_id": "t", "question_group_id": "q", "step_count": 2}
    run = {
        "samples": [sample],
        "methods": {
            method["id"]: [
                {
                    "sample_id": "s",
                    "status": "completed",
                    "diagnosis": {"has_error": False, "first_error_step": None},
                }
            ]
            for method in METHODS
        },
    }
    audit = {
        "processbench_labels": [
            {"trajectory_id": "t", "label": -1},
            {"trajectory_id": "excluded", "label": "invalid-source-row"},
        ]
    }

    report = analyze_diagnoses(run, audit)

    assert report["trajectory_count"] == 1
    assert report["methods"][METHODS[0]["id"]]["confusion"]["tn"] == 1


def test_api_failure_records_raw_response_usage_and_incomplete_status(
    tmp_path: Path, diagnosis_endpoint: str
):
    """Retain non-success HTTP evidence for each failed method without external calls."""
    DiagnosisEndpoint.status_code = 503
    config = RunConfig.model_validate(
        {
            "model": {
                "base_url": diagnosis_endpoint,
                "model": "local-test-model",
                "capabilities": {"supported_parameters": ["max_tokens"], "required_parameters": ["max_tokens"]},
                "parameters": {"max_tokens": 64},
            },
            "budget": {"max_attempts_per_call": 1, "max_reserved_output_tokens": 512},
        }
    )
    manifest = {
        "study": "ms3",
        "manifest_id": "failure-fixture",
        "fingerprint": "failure-digest",
        "samples": [
            {
                "sample_id": "s",
                "source": "processbench",
                "trajectory_id": "t",
                "question_group_id": "q",
                "problem": "Problem",
                "steps": ["step one", "step two"],
            }
        ],
    }
    output = tmp_path / "failed-run"

    run = run_diagnosis(config, manifest, output)

    assert run["status"] == "incomplete"
    assert len(DiagnosisEndpoint.requests) == 3
    for method in METHODS:
        result = run["methods"][method["id"]][0]
        assert result["status"] == "api_failure"
        assert result["usage"] == {"input_tokens": 11, "output_tokens": 7, "status": "known"}
        assert result["response_ref"].endswith(".bin")
        assert result["elapsed_ms"] is not None
        assert result["failure_classification"] == "api_failure"
        assert (output / result["response_ref"]).is_file()

def test_shared_output_budget_failure_is_recorded_without_extra_dispatch(
    tmp_path: Path, diagnosis_endpoint: str
):
    """Record exhausted shared budget per method and do not exceed it with sends."""
    config = RunConfig.model_validate(
        {
            "model": {
                "base_url": diagnosis_endpoint,
                "model": "local-test-model",
                "capabilities": {"supported_parameters": ["max_tokens"], "required_parameters": ["max_tokens"]},
                "parameters": {"max_tokens": 64},
            },
            "budget": {"max_attempts_per_call": 1, "max_reserved_output_tokens": 64},
        }
    )
    manifest = {
        "study": "ms3",
        "manifest_id": "budget-fixture",
        "fingerprint": "budget-digest",
        "samples": [
            {
                "sample_id": "s",
                "source": "processbench",
                "trajectory_id": "t",
                "question_group_id": "q",
                "problem": "Problem",
                "steps": ["step one", "step two"],
            }
        ],
    }

    run = run_diagnosis(config, manifest, tmp_path / "budget-run")

    assert len(DiagnosisEndpoint.requests) == 1
    assert run["status"] == "incomplete"
    statuses = [run["methods"][method["id"]][0]["status"] for method in METHODS]
    assert statuses == ["completed", "run_failure", "run_failure"]
    assert [run["methods"][method["id"]][0]["attempts"] for method in METHODS] == [1, 0, 0]


def test_diagnosis_rejects_tool_parameters():
    """Never allow an endpoint's configured tool options into a diagnosis run."""
    from ricl_experiment.motivation.ms3.protocol import build_diagnosis

    with pytest.raises(ValueError, match="must not configure tools"):
        build_diagnosis(
            METHODS[0],
            {"sample_id": "trace", "problem": "Problem", "steps": ["one", "two"]},
            "test-model",
            {"max_tokens": 64, "tools": "enabled"},
        )


class FailingTransport:
    """Raise a local transport error without touching a network endpoint."""

    def send(self, request_body):
        raise OSError("local test transport failed")


def test_runtime_failure_is_distinct_from_api_failure(tmp_path: Path):
    """Classify a local transport failure separately and preserve its attempt count."""
    config = RunConfig.model_validate(
        {
            "model": {
                "base_url": "http://127.0.0.1:1/v1",
                "model": "local-test-model",
                "capabilities": {"supported_parameters": ["max_tokens"], "required_parameters": ["max_tokens"]},
                "parameters": {"max_tokens": 64},
            },
            "budget": {"max_attempts_per_call": 1, "max_reserved_output_tokens": 512},
        }
    )
    manifest = {
        "study": "ms3",
        "manifest_id": "runtime-fixture",
        "fingerprint": "runtime-digest",
        "samples": [
            {
                "sample_id": "s",
                "source": "processbench",
                "trajectory_id": "t",
                "question_group_id": "q",
                "problem": "Problem",
                "steps": ["step one", "step two"],
            }
        ],
    }

    run = run_diagnosis(config, manifest, tmp_path / "runtime-run", FailingTransport())

    for method in METHODS:
        result = run["methods"][method["id"]][0]
        assert result["status"] == "run_failure"
        assert result["failure_classification"] == "run_failure"
        assert result["attempts"] == 1
