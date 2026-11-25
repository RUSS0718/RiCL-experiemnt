"""Contract tests for the auditable single-sample CLI and controlled HTTP boundary."""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from ricl_experiment.config import fingerprint, load_config, load_manifest
from ricl_experiment.core.ledger import Ledger
from ricl_experiment.runner import resume_run, start_run


class SimulatedProcessExit(BaseException):
    """Stop dispatch at a durable crash boundary without runner exception handling."""


class ControlledEndpoint(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []
    request_lock = threading.Lock()
    delay_seconds = 0
    status_code = 200
    status_sequence: ClassVar[list[int] | None] = None
    include_usage = True
    artifacts_root = None
    pre_send_attempts: ClassVar[list] = []
    request_seen = threading.Event()
    release_request = threading.Event()
    block_request = False

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with self.request_lock:
            self.requests.append((self.path, dict(self.headers), json.loads(body)))
            request_index = len(self.requests) - 1
            self.request_seen.set()
        if self.artifacts_root is not None:
            databases = list((self.artifacts_root / "motivation").glob("*/run.sqlite3"))
            if databases:
                with sqlite3.connect(databases[0]) as db:
                    self.pre_send_attempts.append(
                        db.execute(
                            "SELECT status, reserved_output_tokens FROM attempts "
                            "ORDER BY attempt_number DESC LIMIT 1"
                        ).fetchone()
                    )
        if self.block_request:
            self.release_request.wait(timeout=30)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        payload = {
            "id": "chatcmpl-controlled-1",
            "object": "chat.completion",
            "created": 1,
            "model": "controlled-model",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}
            ],
        }
        if self.include_usage:
            payload["usage"] = {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            }
        encoded = json.dumps(payload).encode()
        status_code = (
            self.status_sequence[request_index]
            if self.status_sequence and request_index < len(self.status_sequence)
            else self.status_code
        )
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except OSError:
            pass

    def log_message(self, format, *args):
        return


@pytest.fixture
def controlled_endpoint():
    ControlledEndpoint.requests = []
    ControlledEndpoint.delay_seconds = 0
    ControlledEndpoint.status_code = 200
    ControlledEndpoint.status_sequence = None
    ControlledEndpoint.include_usage = True
    ControlledEndpoint.artifacts_root = None
    ControlledEndpoint.pre_send_attempts = []
    ControlledEndpoint.request_seen = threading.Event()
    ControlledEndpoint.release_request = threading.Event()
    ControlledEndpoint.block_request = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), ControlledEndpoint)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        ControlledEndpoint.release_request.set()
        server.shutdown()
        thread.join()
        server.server_close()


def write_inputs(tmp_path: Path, endpoint: str, *, timeout: float = 2) -> tuple[Path, Path, Path]:
    config = tmp_path / "model.toml"
    config.write_text(
        "\n".join(
            [
                "[model]",
                f'base_url = "{endpoint}"',
                'model = "controlled-model"',
                'api_key_env = "CONTROLLED_API_KEY"',
                f"timeout_seconds = {timeout}",
                "",
                "[model.capabilities]",
                'supported_parameters = ["max_tokens", "temperature"]',
                'required_parameters = ["max_tokens"]',
                "",
                "[model.parameters]",
                "max_tokens = 32",
                "temperature = 0",
                "",
                "[budget]",
                "max_attempts_per_call = 2",
                "max_reserved_output_tokens = 64",
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
                "manifest_id": "controlled-one-sample",
                "dataset_version": "fixture-v1",
                "samples": [{"sample_id": "sample-1", "prompt": "What is 6 times 7?"}],
            }
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    return config, manifest, artifacts


def invoke_cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ricl_experiment", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def start_command(config: Path, manifest: Path, artifacts: Path, env: dict[str, str]):
    return invoke_cli(
        "run",
        "--study",
        "ms2",
        "--phase",
        "pilot",
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--artifacts-root",
        str(artifacts),
        env=env,
    )


def test_run_dispatches_and_persists_auditable_single_sample(tmp_path, controlled_endpoint):
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    ControlledEndpoint.artifacts_root = artifacts
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}

    result = start_command(config, manifest, artifacts, env)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    run_id = result.stdout.strip().removeprefix("run_id=")
    assert len(run_id) == 32
    assert len(ControlledEndpoint.requests) == 1
    assert ControlledEndpoint.pre_send_attempts == [("pending", 32)]
    path, headers, body = ControlledEndpoint.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer local-test-secret"
    assert body == {
        "model": "controlled-model",
        "messages": [{"role": "user", "content": "What is 6 times 7?"}],
        "max_tokens": 32,
        "temperature": 0,
    }

    run_dir = artifacts / "motivation" / run_id
    metadata_text = (run_dir / "metadata.json").read_text()
    metadata = json.loads(metadata_text)
    assert metadata["config_fingerprint"]
    assert metadata["manifest_fingerprint"]
    assert metadata["prompt_fingerprint"]
    assert metadata["status"] == "completed"
    assert metadata["request_summary"]["roles"] == ["user"]
    assert metadata["effective_parameters"] == {"max_tokens": 32, "temperature": 0}
    assert metadata["experiment_commit_sha"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "local-test-secret" not in metadata_text
    assert "What is 6 times 7?" not in metadata_text

    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        call = db.execute(
            "SELECT request_fingerprint, response_ref, owner_attempt_id FROM logical_calls"
        ).fetchone()
        attempt = db.execute(
            "SELECT status, reserved_output_tokens, input_tokens, output_tokens, effective_parameters "
            "FROM attempts"
        ).fetchone()
        analysis_count = db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
    assert call is not None and call[0] and call[1].startswith("responses/") and call[2]
    assert attempt[:4] == ("completed", 32, 7, 3)
    assert json.loads(attempt[4]) == {"max_tokens": 32, "temperature": 0}
    assert analysis_count == 1
    raw_response = json.loads((run_dir / call[1]).read_bytes())
    assert raw_response["choices"][0]["message"]["content"] == "42"


def test_resume_reuses_the_complete_response_without_a_second_send(tmp_path, controlled_endpoint):
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}
    started = start_command(config, manifest, artifacts, env)
    run_id = started.stdout.strip().removeprefix("run_id=")

    resumed = invoke_cli(
        "resume",
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--artifacts-root",
        str(artifacts),
        env=env,
    )

    assert resumed.returncode == 0, resumed.stderr
    assert resumed.stdout.strip() == f"run_id={run_id}"
    assert len(ControlledEndpoint.requests) == 1
    with sqlite3.connect(artifacts / "motivation" / run_id / "run.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0] == 1


def test_resume_rejects_changed_manifest_before_network_dispatch(tmp_path, controlled_endpoint):
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}
    started = start_command(config, manifest, artifacts, env)
    run_id = started.stdout.strip().removeprefix("run_id=")
    data = json.loads(manifest.read_text())
    data["samples"][0]["prompt"] = "Different prompt"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    resumed = invoke_cli(
        "resume",
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--artifacts-root",
        str(artifacts),
        env=env,
    )

    assert resumed.returncode == 2
    assert "manifest_fingerprint" in resumed.stderr
    assert len(ControlledEndpoint.requests) == 1


def test_unsupported_required_parameter_fails_before_send(tmp_path, controlled_endpoint):
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    config.write_text(config.read_text().replace('"max_tokens", "temperature"', '"temperature"'))
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}

    result = start_command(config, manifest, artifacts, env)

    assert result.returncode == 2
    assert "unsupported configured model parameters: max_tokens" in result.stderr
    assert ControlledEndpoint.requests == []
    assert not (artifacts / "motivation").exists()


def test_timeout_has_bounded_retries_and_persists_unknown_reservations(
    tmp_path, controlled_endpoint
):
    ControlledEndpoint.delay_seconds = 0.15
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint, timeout=0.03)
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}

    result = start_command(config, manifest, artifacts, env)

    assert result.returncode == 2
    assert "timed out" in result.stderr
    assert "run_id=" in result.stderr
    run_id = next((artifacts / "motivation").iterdir()).name
    assert f"run_id={run_id}" in result.stderr
    assert len(ControlledEndpoint.requests) == 2
    with sqlite3.connect(artifacts / "motivation" / run_id / "run.sqlite3") as db:
        attempts = db.execute(
            "SELECT status, usage_status, reserved_output_tokens FROM attempts ORDER BY attempt_number"
        ).fetchall()
        call_status = db.execute("SELECT status FROM logical_calls").fetchone()[0]
    assert attempts == [("unknown", "unknown", 32), ("unknown", "unknown", 32)]
    assert call_status == "unknown"
    metadata = json.loads((artifacts / "motivation" / run_id / "metadata.json").read_text())
    assert metadata["usage"] == {
        "known_output_tokens": 0,
        "reserved_output_tokens": 64,
        "unknown_usage_attempts": 2,
    }


def test_resume_retries_a_persisted_unconfirmed_attempt(tmp_path, controlled_endpoint):
    config_path, manifest_path, artifacts = write_inputs(tmp_path, controlled_endpoint)
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}
    config = load_config(config_path)
    manifest = load_manifest(manifest_path)
    run_id = uuid.uuid4().hex
    run_dir = artifacts / "motivation" / run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "study": "ms2",
        "phase": "pilot",
        "config_fingerprint": fingerprint(config.redacted()),
        "manifest_fingerprint": fingerprint(manifest.model_dump(mode="json")),
        "prompt_fingerprint": fingerprint([sample.prompt for sample in manifest.samples]),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    call_id = fingerprint(
        {
            "run_id": run_id,
            "study": "ms2",
            "phase": "pilot",
            "sample_id": "sample-1",
            "sample_index": 0,
        }
    )
    request = {
        "model": config.model.model,
        "messages": [{"role": "user", "content": "What is 6 times 7?"}],
        **config.model.parameters,
    }
    ledger = Ledger(run_dir / "run.sqlite3", config.budget.max_reserved_output_tokens)
    ledger.ensure_call(call_id, "sample-1", fingerprint(request))
    ledger.start_attempt(call_id, fingerprint(request), config.model.parameters, 32, 2)
    ledger.close()

    resumed = invoke_cli(
        "resume",
        "--run-id",
        run_id,
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--artifacts-root",
        str(artifacts),
        env=env,
    )

    assert resumed.returncode == 0, resumed.stderr
    assert len(ControlledEndpoint.requests) == 1
    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        attempts = db.execute(
            "SELECT status, duplicate_billing_risk FROM attempts ORDER BY attempt_number"
        ).fetchall()
        assert db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0] == 1
    assert attempts == [("unknown", 1), ("completed", 0)]


def test_one_scheduler_owns_a_run_at_a_time(tmp_path, controlled_endpoint):
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint, timeout=35)
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}
    ControlledEndpoint.block_request = True
    first = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ricl_experiment",
            "run",
            "--study",
            "ms2",
            "--phase",
            "pilot",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--artifacts-root",
            str(artifacts),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert ControlledEndpoint.request_seen.wait(timeout=20)
        run_id = next((artifacts / "motivation").iterdir()).name
        second = invoke_cli(
            "resume",
            "--run-id",
            run_id,
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--artifacts-root",
            str(artifacts),
            env=env,
        )
        assert second.returncode == 2
        assert "another scheduler already owns this run_id" in second.stderr
        assert len(ControlledEndpoint.requests) == 1
    finally:
        ControlledEndpoint.release_request.set()
        first_stdout, first_stderr = first.communicate(timeout=5)
    assert first.returncode == 0, first_stderr
    assert f"run_id={run_id}" in first_stdout


def test_response_without_usage_keeps_attempt_reservation(tmp_path, controlled_endpoint):
    ControlledEndpoint.include_usage = False
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}

    result = start_command(config, manifest, artifacts, env)

    assert result.returncode == 0, result.stderr
    run_id = result.stdout.strip().removeprefix("run_id=")
    with sqlite3.connect(artifacts / "motivation" / run_id / "run.sqlite3") as db:
        attempt = db.execute(
            "SELECT status, usage_status, reserved_output_tokens, output_tokens FROM attempts"
        ).fetchone()
    summary = json.loads((artifacts / "motivation" / run_id / "metadata.json").read_text())["usage"]
    assert attempt == ("completed", "unknown", 32, None)
    assert summary == {
        "known_output_tokens": 0,
        "reserved_output_tokens": 32,
        "unknown_usage_attempts": 1,
    }


def test_config_drift_is_rejected_before_network_dispatch(tmp_path, controlled_endpoint):
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}
    started = start_command(config, manifest, artifacts, env)
    run_id = started.stdout.strip().removeprefix("run_id=")
    config.write_text(config.read_text().replace("temperature = 0", "temperature = 0.1"))

    resumed = invoke_cli(
        "resume",
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--artifacts-root",
        str(artifacts),
        env=env,
    )

    assert resumed.returncode == 2
    assert "config_fingerprint" in resumed.stderr
    assert len(ControlledEndpoint.requests) == 1


def test_retry_records_error_response_usage_and_one_result(tmp_path, controlled_endpoint):
    ControlledEndpoint.status_sequence = [503, 200]
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    ControlledEndpoint.artifacts_root = artifacts
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}

    result = start_command(config, manifest, artifacts, env)

    assert result.returncode == 0, result.stderr
    run_id = result.stdout.strip().removeprefix("run_id=")
    run_dir = artifacts / "motivation" / run_id
    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        attempts = db.execute(
            "SELECT status, duplicate_billing_risk, usage_status, input_tokens, output_tokens, "
            "response_ref FROM attempts ORDER BY attempt_number"
        ).fetchall()
        assert db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0] == 1
    assert [row[:5] for row in attempts] == [
        ("unknown", 1, "known", 7, 3),
        ("completed", 0, "known", 7, 3),
    ]
    assert json.loads((run_dir / attempts[0][5]).read_bytes())["choices"][0]["message"]["content"] == "42"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["usage"]["known_output_tokens"] == 6


def test_rejected_answer_field_never_reaches_generation_request(tmp_path, controlled_endpoint):
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    data = json.loads(manifest.read_text())
    data["samples"][0]["answer"] = "42"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}

    result = start_command(config, manifest, artifacts, env)

    assert result.returncode == 2
    assert "answer" in result.stderr
    assert ControlledEndpoint.requests == []


def test_unknown_usage_reservation_prevents_dispatch_over_budget(tmp_path, controlled_endpoint):
    ControlledEndpoint.delay_seconds = 0.15
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint, timeout=0.03)
    config.write_text(
        config.read_text().replace("max_reserved_output_tokens = 64", "max_reserved_output_tokens = 32")
    )
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}

    result = start_command(config, manifest, artifacts, env)

    assert result.returncode == 2
    assert "output-token budget exhausted" in result.stderr
    run_id = next((artifacts / "motivation").iterdir()).name
    assert f"run_id={run_id}" in result.stderr
    assert len(ControlledEndpoint.requests) == 1
    with sqlite3.connect(artifacts / "motivation" / run_id / "run.sqlite3") as db:
        attempts = db.execute(
            "SELECT status, reserved_output_tokens FROM attempts ORDER BY attempt_number"
        ).fetchall()
    assert attempts == [("unknown", 32)]


def test_permanent_http_failure_is_audited_and_not_retried_on_resume(tmp_path, controlled_endpoint):
    ControlledEndpoint.status_code = 400
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}

    started = start_command(config, manifest, artifacts, env)

    assert started.returncode == 2
    assert "run_id=" in started.stderr
    run_id = next((artifacts / "motivation").iterdir()).name
    run_dir = artifacts / "motivation" / run_id
    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        attempt = db.execute(
            "SELECT status, usage_status, input_tokens, output_tokens, response_ref FROM attempts"
        ).fetchone()
        assert db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0] == 0
    assert attempt[:4] == ("failed", "known", 7, 3)
    assert json.loads((run_dir / attempt[4]).read_bytes())["choices"][0]["message"]["content"] == "42"

    resumed = invoke_cli(
        "resume",
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--artifacts-root",
        str(artifacts),
        env=env,
    )

    assert resumed.returncode == 2
    assert "permanent failed attempt" in resumed.stderr
    assert len(ControlledEndpoint.requests) == 1




def _start_crashed_run(tmp_path, endpoint, monkeypatch, method, *, commit_before_exit):
    """Create a run and stop at a simulated durable crash boundary."""
    config_path, manifest_path, artifacts = write_inputs(tmp_path, endpoint)
    monkeypatch.setenv("CONTROLLED_API_KEY", "local-test-secret")
    original = getattr(Ledger, method)

    def exit_at_durable_boundary(self, *args, **kwargs):
        if commit_before_exit:
            original(self, *args, **kwargs)
        raise SimulatedProcessExit

    with monkeypatch.context() as patch:
        patch.setattr(Ledger, method, exit_at_durable_boundary)
        with pytest.raises(SimulatedProcessExit):
            start_run("ms2", "pilot", config_path, manifest_path, artifacts)
    return config_path, manifest_path, artifacts, next((artifacts / "motivation").iterdir())


@pytest.mark.parametrize(
    ("status_code", "attempt_status", "billing_risk"),
    [(429, "retryable", 0), (503, "unknown", 1)],
)
def test_resume_retries_transient_http_failure_after_process_exit(
    tmp_path, controlled_endpoint, monkeypatch, status_code, attempt_status, billing_risk
):
    ControlledEndpoint.status_sequence = [status_code, 200]
    config, manifest, artifacts, run_dir = _start_crashed_run(
        tmp_path, controlled_endpoint, monkeypatch, "record_failure", commit_before_exit=True
    )

    resume_run(run_dir.name, config, manifest, artifacts)

    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        attempts = db.execute(
            "SELECT attempt_id, status, duplicate_billing_risk, response_ref "
            "FROM attempts ORDER BY attempt_number"
        ).fetchall()
        result = db.execute(
            "SELECT attempt_id, response_ref FROM analysis_results WHERE logical_call_id = "
            "(SELECT logical_call_id FROM attempts LIMIT 1)"
        ).fetchone()
    assert attempts[0][1:3] == (attempt_status, billing_risk)
    assert attempts[1][1:3] == ("completed", 0)
    assert result == attempts[1][::3]
    assert len(ControlledEndpoint.requests) == 2


def test_resume_claims_complete_response_saved_before_ledger_commit(
    tmp_path, controlled_endpoint, monkeypatch
):
    config, manifest, artifacts, run_dir = _start_crashed_run(
        tmp_path, controlled_endpoint, monkeypatch, "record_response", commit_before_exit=False
    )
    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        attempt_id, status, response_ref = db.execute(
            "SELECT attempt_id, status, response_ref FROM attempts"
        ).fetchone()
    expected_ref = f"responses/{attempt_id}.json"
    assert (run_dir / expected_ref).is_file() and status == "pending" and response_ref is None

    resume_run(run_dir.name, config, manifest, artifacts)

    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
        attempt = db.execute(
            "SELECT status, usage_status, input_tokens, output_tokens, response_ref FROM attempts"
        ).fetchone()
        call = db.execute("SELECT owner_attempt_id, response_ref FROM logical_calls").fetchone()
        result = db.execute("SELECT attempt_id, response_ref FROM analysis_results").fetchone()
    assert attempt == ("completed", "known", 7, 3, expected_ref)
    assert call == result == (attempt_id, expected_ref)
    assert len(ControlledEndpoint.requests) == 1


@pytest.mark.parametrize("artifact_state", ["incomplete", "invalid_shape", "unmatched"])
def test_resume_audits_and_retries_incomplete_or_unmatched_response_files(
    tmp_path, controlled_endpoint, monkeypatch, artifact_state
):
    ControlledEndpoint.status_sequence = [200, 200]
    config, manifest, artifacts, run_dir = _start_crashed_run(
        tmp_path, controlled_endpoint, monkeypatch, "record_response", commit_before_exit=False
    )
    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        attempt_id = db.execute("SELECT attempt_id FROM attempts").fetchone()[0]
    expected_file = run_dir / "responses" / f"{attempt_id}.json"
    unrelated_file = run_dir / "responses" / "unrelated.json"
    if artifact_state == "incomplete":
        expected_file.write_text('{"choices":', encoding="utf-8")
    elif artifact_state == "invalid_shape":
        expected_file.write_text('{"choices": []}', encoding="utf-8")
    if artifact_state != "unmatched":
        unrelated_file.write_text(
            json.dumps({"choices": [{"message": {"content": "unrelated"}}]}),
            encoding="utf-8",
        )
    else:
        expected_file.replace(unrelated_file)

    resume_run(run_dir.name, config, manifest, artifacts)

    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        attempts = db.execute(
            "SELECT attempt_id, status, usage_status, reserved_output_tokens, output_tokens, "
            "response_ref, error FROM attempts ORDER BY attempt_number"
        ).fetchall()
        owner = db.execute("SELECT owner_attempt_id, response_ref FROM logical_calls").fetchone()
        result_count = db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
    assert attempts[0][0:3] == (attempt_id, "unknown", "unknown")
    assert attempts[0][5] == (
        f"responses/{attempt_id}.json" if artifact_state != "unmatched" else None
    )
    if artifact_state != "unmatched":
        assert "invalid persisted response" in attempts[0][6]
    assert attempts[1][1:5] == ("completed", "known", 32, 3)
    assert owner == (attempts[1][0], attempts[1][5])
    assert result_count == 1 and len(ControlledEndpoint.requests) == 2


@pytest.mark.parametrize(
    ("max_attempts", "budget", "error_text"),
    [(1, 64, "attempt limit"), (2, 32, "output-token budget exhausted")],
)
def test_transient_failure_resume_remains_within_attempt_and_budget_limits(
    tmp_path, controlled_endpoint, max_attempts, budget, error_text
):
    ControlledEndpoint.status_code = 429
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    config.write_text(
        config.read_text(encoding="utf-8")
        .replace("max_attempts_per_call = 2", f"max_attempts_per_call = {max_attempts}")
        .replace("max_reserved_output_tokens = 64", f"max_reserved_output_tokens = {budget}"),
        encoding="utf-8",
    )
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}
    started = start_command(config, manifest, artifacts, env)
    assert started.returncode == 2
    run_id = next((artifacts / "motivation").iterdir()).name
    resumed = invoke_cli(
        "resume", "--run-id", run_id, "--config", str(config), "--manifest", str(manifest),
        "--artifacts-root", str(artifacts), env=env,
    )
    assert resumed.returncode == 2 and error_text in resumed.stderr
    with sqlite3.connect(artifacts / "motivation" / run_id / "run.sqlite3") as db:
        assert db.execute("SELECT status FROM attempts").fetchall() == [("retryable",)]
    assert len(ControlledEndpoint.requests) == 1


def test_experiment_commit_sha_marks_git_unavailable_explicitly(monkeypatch):
    from ricl_experiment import runner

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git executable is unavailable")

    monkeypatch.setattr(runner.subprocess, "run", missing_git)
    assert runner._experiment_commit_sha() == "unavailable"


def test_resume_keeps_permanent_http_failure_after_an_earlier_transient(
    tmp_path, controlled_endpoint
):
    ControlledEndpoint.status_sequence = [503, 400, 200]
    config, manifest, artifacts = write_inputs(tmp_path, controlled_endpoint)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "max_attempts_per_call = 2", "max_attempts_per_call = 3"
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "CONTROLLED_API_KEY": "local-test-secret"}
    started = start_command(config, manifest, artifacts, env)
    assert started.returncode == 2
    run_id = next((artifacts / "motivation").iterdir()).name

    resumed = invoke_cli(
        "resume", "--run-id", run_id, "--config", str(config), "--manifest", str(manifest),
        "--artifacts-root", str(artifacts), env=env,
    )

    assert resumed.returncode == 2 and "permanent failed attempt" in resumed.stderr
    assert len(ControlledEndpoint.requests) == 2
