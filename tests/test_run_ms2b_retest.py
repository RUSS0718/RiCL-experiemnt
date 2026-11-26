"""Offline acceptance tests for the frozen MS2-B retest dispatcher."""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Self

import pytest

from ricl_experiment.config import RunConfig, fingerprint
from ricl_experiment.core import model
from ricl_experiment.core.ledger import Ledger
from ricl_experiment.core.model import ModelDispatchError, dispatch_request

_runner_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ms2b_retest.py"
_runner_spec = spec_from_file_location("run_ms2b_retest", _runner_path)
if _runner_spec is None or _runner_spec.loader is None:
    raise RuntimeError("could not load MS2-B retest runner")
runner = module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def isolate_dispatch_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "DISPATCH_GUARD_DIR", tmp_path / "dispatch-guards")
    monkeypatch.setattr(runner, "PRIOR_RUN_DIR", tmp_path / "original-run")
    monkeypatch.setattr(runner, "_has_dispatch_time", lambda config: True)


def _frozen_inputs(directory: Path) -> Path:
    """Build the same frozen JSON contract as the preparation script, without I/O to providers."""
    directory.mkdir()
    samples = []
    pairs = []
    requests = []
    for index in range(64):
        group = f"group-{index:02d}"
        left_id = f"prefix-{index:02d}-a"
        right_id = f"prefix-{index:02d}-b"
        for sample_id in (left_id, right_id):
            sample = {
                "sample_id": sample_id,
                "question_group_id": group,
                "source_stratum": "pb:math",
                "source": "processbench",
                "problem": f"Problem {index}: compute {index} + 1.",
                "checkpoint": f"Prefix {sample_id}: start with {index} + 1.",
            }
            samples.append(sample)
            body = runner.build_request_body(sample)
            requests.append(
                {
                    "sample_id": sample_id,
                    "question_group_id": group,
                    "source_stratum": "pb:math",
                    "request_fingerprint": fingerprint(body),
                }
            )
        pairs.append(
            {
                "group": group,
                "left": {"id": left_id},
                "right": {"id": right_id},
            }
        )
    manifest = {
        "schema_version": 1,
        "manifest_id": runner.MANIFEST_ID,
        "study": "ms2",
        "phase": "adjustment_validation",
        "sampling": {"planned_groups": 64},
        "samples": samples,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    plan = {
        "manifest_id": manifest["manifest_id"],
        "manifest_fingerprint": manifest["fingerprint"],
        "model": runner.MODEL,
        "parameters": runner.PARAMETERS,
        "max_attempts_per_logical_call": 1,
        "logical_calls": 128,
        "requests": requests,
        "status": "prepared_not_dispatched",
    }
    audit = {
        "manifest_id": manifest["manifest_id"],
        "manifest_fingerprint": manifest["fingerprint"],
        "pairs": pairs,
    }
    for name, value in (
        ("manifest.json", manifest),
        ("request-plan.json", plan),
        ("restricted/audit.json", audit),
    ):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return directory


def _config(path: Path) -> Path:
    """Write a provider-shaped config whose transport is replaced in offline tests."""
    path.write_text(
        """[model]
base_url = "https://code.aigcpro.org/v1"
model = "deepseek-v4.1-flash"

[model.capabilities]
supported_parameters = ["max_tokens"]
required_parameters = ["max_tokens"]

[model.parameters]
max_tokens = 2048

[budget]
max_attempts_per_call = 1
max_reserved_output_tokens = 2048
""",
        encoding="utf-8",
    )
    return path


def test_preflight_rebuilds_and_fingerprints_all_128_exact_parameter_bodies(tmp_path: Path) -> None:
    frozen = _frozen_inputs(tmp_path / "frozen")

    requests = runner.build_frozen_requests(frozen)

    assert len(requests) == 128
    assert len({item["question_group_id"] for item in requests}) == 64
    assert runner.PARAMETERS == {
        "max_tokens": 2048,
        "stream": False,
        "thinking": {"type": "enabled"},
        "response_format": {"type": "json_object"},
    }
    assert all(item["request_fingerprint"] == fingerprint(item["body"]) for item in requests)
    assert all(item["body"]["model"] == "deepseek-v4.1-flash" for item in requests)
    assert all(item["body"]["max_tokens"] == 2048 for item in requests)
    assert all(item["body"]["stream"] is False for item in requests)
    assert all(item["body"]["thinking"] == {"type": "enabled"} for item in requests)
    assert all(item["body"]["response_format"] == {"type": "json_object"} for item in requests)
    assert all(
        "Rate only the displayed prefix." in item["body"]["messages"][0]["content"]
        for item in requests
    )


def test_default_run_persists_offline_preflight_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen_inputs(tmp_path / "frozen")
    output = tmp_path / "preflight"

    def forbidden_dispatch(*args: object, **kwargs: object) -> str:
        pytest.fail("offline preflight must not dispatch")

    monkeypatch.setattr(runner, "dispatch_request", forbidden_dispatch)
    record = runner.run(frozen, output)

    assert record["status"] == "preflight_complete_not_dispatched"
    assert len(record["calls"]) == 128
    assert (
        len((output / "restricted/requests.jsonl").read_text(encoding="utf-8").splitlines()) == 128
    )
    assert not (output / "run.sqlite3").exists()


def test_preflight_fails_closed_on_fingerprint_mismatch(tmp_path: Path) -> None:
    frozen = _frozen_inputs(tmp_path / "frozen")
    plan_path = frozen / "request-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["requests"][7]["request_fingerprint"] = "0" * 64
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        runner.build_frozen_requests(frozen)


def test_reservation_counts_ms3b_plus_old_unknown_against_300() -> None:
    assert runner.request_reservation(160, 128) == {
        "cap": 300,
        "known_ms3b_attempts": 160,
        "reserved_old_unknown_attempts": 12,
        "this_run_attempts": 128,
        "total_reserved_attempts": 300,
        "remaining_attempts": 0,
    }
    with pytest.raises(ValueError, match="300"):
        runner.request_reservation(161, 128)


def test_live_runner_stops_route_after_first_error_and_record_precedes_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen_inputs(tmp_path / "frozen")
    output = tmp_path / "run"
    config = _config(tmp_path / "model.toml")
    completion = tmp_path / "ms3b-completion.json"
    ms3b_run_dir = tmp_path / "ms3b-run"
    ms3b_run_dir.mkdir()
    Ledger(ms3b_run_dir / "run.sqlite3", 1).close()
    completion.write_text(
        json.dumps(
            {
                "status": "completed",
                "scope": "MS3-B confirmatory",
                "model": runner.MODEL,
                "run_id": "ms3b-run",
                "run_dir": str(ms3b_run_dir),
            }
        ),
        encoding="utf-8",
    )
    sent: list[str] = []

    def fail_first(*args: object, **kwargs: object) -> str:
        call_id = str(args[3])
        sent.append(call_id)
        record = json.loads((output / "dispatch-record.json").read_text(encoding="utf-8"))
        assert record["status"] == "dispatching"
        assert (output / "restricted/requests.jsonl").is_file()
        raise ModelDispatchError("simulated offline failure")

    monkeypatch.setattr(runner, "dispatch_request", fail_first)

    record = runner.run(
        frozen, output, dispatch=True, config_path=config, ms3b_completion_path=completion
    )

    assert record["status"] == "route_stopped"
    assert len(sent) == 1
    assert record["calls"][0]["status"] == "error"
    assert all(call["status"] == "not_started" for call in record["calls"][1:])


def test_live_runner_rejects_unapproved_provider_before_creating_output(tmp_path: Path) -> None:
    frozen = _frozen_inputs(tmp_path / "frozen")
    output = tmp_path / "run"
    config = _config(tmp_path / "model.toml")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "https://code.aigcpro.org/v1", "https://offline.invalid/v1"
        ),
        encoding="utf-8",
    )
    completion = tmp_path / "ms3b-completion.json"

    with pytest.raises(ValueError, match="approved AIGCPro route"):
        runner.run(
            frozen, output, dispatch=True, config_path=config, ms3b_completion_path=completion
        )

    assert not output.exists()


def test_live_runner_requires_completed_ms3b_record(tmp_path: Path) -> None:
    frozen = _frozen_inputs(tmp_path / "frozen")
    output = tmp_path / "run"
    config = _config(tmp_path / "model.toml")
    completion = tmp_path / "ms3b-completion.json"
    completion.write_text(
        json.dumps({"status": "running", "scope": "MS3-B confirmatory", "model": runner.MODEL}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must say completed"):
        runner.run(
            frozen, output, dispatch=True, config_path=config, ms3b_completion_path=completion
        )

    assert not output.exists()


def test_dispatch_request_does_not_resend_unknown_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    class FakeResponse:
        status_code = 503
        content = b'{"error":"temporary gateway error"}'
        is_success = False

        @staticmethod
        def json() -> dict[str, str]:
            return {"error": "temporary gateway error"}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def post(self, *args: object, **kwargs: object) -> FakeResponse:
            nonlocal sends
            sends += 1
            return FakeResponse()

    sends = 0
    monkeypatch.setattr(model.httpx, "Client", FakeClient)
    config = RunConfig.model_validate(
        {
            "model": {
                "base_url": "https://offline.invalid/v1",
                "model": "deepseek-v4.1-flash",
                "capabilities": {"supported_parameters": ["max_tokens"]},
                "parameters": {"max_tokens": 2048},
            },
            "budget": {"max_attempts_per_call": 1, "max_reserved_output_tokens": 2048},
        }
    )
    ledger = Ledger(tmp_path / "run.sqlite3", 2048)
    run_dir = tmp_path / "dispatch"

    try:
        for _ in range(2):
            with pytest.raises(ModelDispatchError):
                dispatch_request(
                    config,
                    ledger,
                    run_dir,
                    "stable-logical-call",
                    "sample-1",
                    {"model": "deepseek-v4.1-flash", "messages": []},
                    effective_parameters=runner.PARAMETERS,
                )
    finally:
        ledger.close()

    assert sends == 1


def test_resume_dispatch_sends_only_unstarted_calls_and_guards_frozen_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen_inputs(tmp_path / "frozen")
    output = tmp_path / "run"
    config = _config(tmp_path / "model.toml")
    completion = tmp_path / "ms3b-completion.json"
    ms3b_run_dir = tmp_path / "ms3b-run"
    ms3b_run_dir.mkdir()
    Ledger(ms3b_run_dir / "run.sqlite3", 1).close()
    completion.write_text(
        json.dumps(
            {
                "status": "completed",
                "scope": "MS3-B confirmatory",
                "model": runner.MODEL,
                "run_id": "ms3b-run",
                "run_dir": str(ms3b_run_dir),
            }
        ),
        encoding="utf-8",
    )
    first_call = runner.build_frozen_requests(frozen)[0]
    first_attempts: list[str] = []

    def time_out_first(
        config: RunConfig,
        ledger: Ledger,
        run_dir: Path,
        call_id: str,
        sample_id: str,
        request_body: dict[str, object],
        *,
        effective_parameters: dict[str, object],
    ) -> str:
        first_attempts.append(call_id)
        request_fingerprint = fingerprint(request_body)
        ledger.ensure_call(call_id, sample_id, request_fingerprint, request_body=request_body)
        attempt_id = ledger.start_attempt(
            call_id, request_fingerprint, effective_parameters, 2048, 1
        )
        ledger.record_failure(attempt_id, "unknown", "ConnectTimeout", usage=None)
        raise ModelDispatchError("timed out with unknown acceptance")

    monkeypatch.setattr(runner, "dispatch_request", time_out_first)
    stopped = runner.run(
        frozen, output, dispatch=True, config_path=config, ms3b_completion_path=completion
    )
    assert stopped["status"] == "route_stopped"
    assert first_attempts == [first_call["call_id"]]

    resumed_calls: list[str] = []

    def complete_remaining(
        config: RunConfig,
        ledger: Ledger,
        run_dir: Path,
        call_id: str,
        sample_id: str,
        request_body: dict[str, object],
        *,
        effective_parameters: dict[str, object],
    ) -> str:
        resumed_calls.append(call_id)
        request_fingerprint = fingerprint(request_body)
        ledger.ensure_call(call_id, sample_id, request_fingerprint, request_body=request_body)
        attempt_id = ledger.start_attempt(
            call_id, request_fingerprint, effective_parameters, 2048, 1
        )
        response_ref = f"responses/{sample_id}.json"
        response = run_dir / response_ref
        response.parent.mkdir(exist_ok=True)
        response.write_text(
            json.dumps(
                {"choices": [{"message": {"content": '{"rating":2}'}, "finish_reason": "stop"}]}
            ),
            encoding="utf-8",
        )
        ledger.record_response(call_id, attempt_id, response_ref, usage=(1, 1, 0))
        return response_ref

    monkeypatch.setattr(runner, "dispatch_request", complete_remaining)
    completed = runner.resume_dispatch(
        frozen, output, config_path=config, ms3b_completion_path=completion
    )

    assert completed["status"] == "completed"
    assert len(resumed_calls) == runner.CALL_COUNT - 1
    assert first_call["call_id"] not in resumed_calls
    assert completed["calls"][0]["status"] == "error"
    assert sum(call["status"] == "completed" for call in completed["calls"]) == 127

    with pytest.raises(ValueError, match="already consumed"):
        runner.run(
            frozen,
            tmp_path / "second-run",
            dispatch=True,
            config_path=config,
            ms3b_completion_path=completion,
        )
