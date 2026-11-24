"""MS3 diagnosis orchestration, contracts, and auditable request artifacts."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...config import RunConfig, canonical_json, fingerprint
from ...core.ledger import Ledger
from .protocol import (
    METHOD_IDS,
    METHODS,
    TEMPLATE_VERSION,
    DiagnosisAPIError,
    DiagnosisRuntimeError,
    DiagnosisSample,
    DiagnosisTransport,
    HttpxDiagnosisTransport,
    build_diagnosis,
    parse_diagnosis,
)


def run_diagnosis(
    config: RunConfig,
    manifest: Mapping[str, Any] | Path,
    output_dir: Path,
    transport: DiagnosisTransport | None = None,
) -> dict[str, Any]:
    """Run the three paired diagnosis calls and write an auditable JSON artifact."""
    manifest_value = _load_mapping(manifest)
    if manifest_value.get("study") != "ms3":
        raise ValueError("MS3 diagnosis requires an MS3 sample manifest")
    manifest_fingerprint = manifest_value.get("fingerprint")
    if not isinstance(manifest_fingerprint, str) or not manifest_fingerprint.strip():
        raise ValueError("MS3 diagnosis manifest requires a fingerprint")
    sample_rows = _sample_rows(manifest_value)
    sample_ids = [row.get("sample_id") for row in sample_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("MS3 manifest sample IDs must be unique")
    if any(row.get("source") != "processbench" for row in sample_rows):
        raise ValueError("MS3 diagnosis requires ProcessBench samples")
    samples = [DiagnosisSample.from_mapping(row) for row in sample_rows]
    requests = {
        (sample.sample_id, method["id"]): build_diagnosis(
            method, sample.public_record(), config.model.model, config.model.parameters
        )
        for sample in samples
        for method in METHODS
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "requests").mkdir()
    (output_dir / "responses").mkdir()
    transport = transport or HttpxDiagnosisTransport(config)
    ledger = Ledger(output_dir / "run.sqlite3", config.budget.max_reserved_output_tokens)
    methods = {method_id: [] for method_id in METHOD_IDS}
    artifact = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "study": "ms3",
        "manifest_id": manifest_value.get("manifest_id"),
        "manifest_fingerprint": manifest_fingerprint,
        "model": config.model.model,
        "effective_parameters": dict(config.model.parameters) | {"tool_choice": "none"},
        "tools_enabled": False,
        "method_template_version": TEMPLATE_VERSION,
        "methods": methods,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "trajectory_id": sample.trajectory_id,
                "question_group_id": sample.question_group_id,
                "step_count": len(sample.steps),
                **(
                    {"source_record_id": sample.source_record_id}
                    if sample.source_record_id is not None
                    else {}
                ),
            }
            for sample in samples
        ],
        "status": "running",
        "usage": ledger.usage_summary(),
    }
    _write_json(output_dir / "run.json", artifact)
    try:
        for sample in samples:
            for method in METHODS:
                methods[method["id"]].append(
                    _run_one_call(
                        config,
                        ledger,
                        output_dir,
                        transport,
                        sample,
                        method,
                        requests[(sample.sample_id, method["id"])],
                    )
                )
                artifact["usage"] = ledger.usage_summary()
                _write_json(output_dir / "run.json", artifact)
        run_status = "completed" if all(
            row["status"] not in {"run_failure", "api_failure"}
            for rows in methods.values()
            for row in rows
        ) else "incomplete"
        ledger.close_run(run_status)
        artifact["status"] = run_status
        artifact["usage"] = ledger.usage_summary()
        _write_json(output_dir / "run.json", artifact)
        return artifact
    finally:
        ledger.close()


def _run_one_call(
    config: RunConfig,
    ledger: Ledger,
    output_dir: Path,
    transport: DiagnosisTransport,
    sample: DiagnosisSample,
    method: Mapping[str, str],
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch and persist one method/trajectory logical call."""
    call_id = fingerprint(
        {
            "study": "ms3",
            "sample_id": sample.sample_id,
            "method": method["id"],
            "request": request_body,
        }
    )
    request_ref = f"requests/{call_id}.json"
    _write_json(output_dir / request_ref, request_body)
    request_summary = {
        "logical_call_id": call_id,
        "method": method["id"],
        "role": "diagnosis",
        "model": config.model.model,
        "effective_parameters": dict(config.model.parameters) | {"tool_choice": "none"},
        "tools_enabled": False,
        "method_template_version": TEMPLATE_VERSION,
        "visible_fields": ["problem", "steps"],
        "input_fingerprint": fingerprint({"problem": sample.problem, "steps": sample.steps}),
        "request_fingerprint": fingerprint(request_body),
        "request_ref": request_ref,
    }
    ledger.ensure_call(call_id, sample.sample_id, request_summary["request_fingerprint"])
    last_error = None
    last_usage = None
    last_response_ref = None
    last_elapsed_ms = None
    attempts = 0
    failure_status = "run_failure"
    failure_classification = "run_failure"
    for _ in range(config.budget.max_attempts_per_call):
        try:
            attempt_id = ledger.start_attempt(
                call_id,
                request_summary["request_fingerprint"],
                dict(request_summary["effective_parameters"]),
                int(config.model.parameters["max_tokens"]),
                config.budget.max_attempts_per_call,
            )
        except ValueError as error:
            last_error = str(error)
            break
        attempts += 1
        try:
            reply = transport.send(request_body)
            response_ref = f"responses/{attempt_id}.json"
            _write_json(output_dir / response_ref, dict(reply.raw_response))
            ledger.record_response(call_id, attempt_id, response_ref, _usage_tuple(reply.usage))
            status, diagnosis, parse_error = parse_diagnosis(reply.content, len(sample.steps))
            failure = {
                "abstained": "abstained",
                "parse_error": "parse_error",
                "invalid_output": "invalid_output",
            }.get(status)
            return _result(
                sample,
                method["id"],
                status,
                diagnosis,
                failure,
                parse_error,
                reply.content,
                reply.usage,
                reply.elapsed_ms,
                request_summary,
                response_ref,
                attempts,
            )
        except DiagnosisRuntimeError as error:
            last_error = f"{type(error).__name__}: {error}"
            failure_status = "run_failure"
            failure_classification = "run_failure"
            last_elapsed_ms = error.elapsed_ms
            ledger.record_failure(attempt_id, "unknown", last_error, usage=None)
        except DiagnosisAPIError as error:
            last_error = f"{type(error).__name__}: {error}"
            last_usage = error.usage
            last_elapsed_ms = error.elapsed_ms
            failure_status = "api_failure"
            failure_classification = "api_failure"
            response_ref = None
            if error.raw_response is not None:
                response_ref = f"responses/{attempt_id}.bin"
                _write_bytes(output_dir / response_ref, error.raw_response)
            attempt_status = "unknown" if error.retryable else "failed"
            last_response_ref = response_ref
            ledger.record_failure(
                attempt_id,
                attempt_status,
                last_error,
                _usage_tuple(error.usage),
                response_ref=response_ref,
            )
            if not error.retryable:
                break
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            last_error = f"{type(error).__name__}: {error}"
            failure_status = "run_failure"
            failure_classification = "run_failure"
            ledger.record_failure(attempt_id, "unknown", last_error, usage=None)
    return _result(
        sample,
        method["id"],
        failure_status,
        None,
        failure_classification,
        last_error or "no attempt completed",
        None,
        last_usage,
        last_elapsed_ms,
        request_summary,
        last_response_ref,
        attempts,
    )


def _result(
    sample: DiagnosisSample,
    method_id: str,
    status: str,
    diagnosis: dict[str, Any] | None,
    failure: str | None,
    error: str | None,
    raw_output: str | None,
    usage: dict[str, int] | None,
    elapsed_ms: int | None,
    request: dict[str, Any],
    response_ref: str | None,
    attempts: int,
) -> dict[str, Any]:
    """Build one JSON-safe per-method audit row."""
    return {
        "sample_id": sample.sample_id,
        "trajectory_id": sample.trajectory_id,
        "question_group_id": sample.question_group_id,
        "method": method_id,
        "status": status,
        "failure_classification": failure,
        "diagnosis": diagnosis,
        "raw_output": raw_output,
        "usage": _usage_record(usage),
        "elapsed_ms": elapsed_ms,
        "request": request,
        "response_ref": response_ref,
        "attempts": attempts,
        "error": error,
    }


def _sample_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract a non-empty sample list from a manifest-like mapping."""
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("MS3 diagnosis manifest must contain a non-empty samples list")
    if any(not isinstance(sample, Mapping) for sample in samples):
        raise ValueError("MS3 diagnosis samples must be objects")
    return samples


def _is_chat_completion(value: Any) -> bool:
    """Recognize the assistant content shape needed by the transport."""
    return bool(
        isinstance(value, dict)
        and isinstance(value.get("choices"), list)
        and value["choices"]
        and isinstance(value["choices"][0], dict)
        and isinstance(value["choices"][0].get("message"), dict)
        and isinstance(value["choices"][0]["message"].get("content"), str)
    )


def _usage(value: Any) -> dict[str, int] | None:
    """Extract non-negative prompt and completion token counts when both exist."""
    if not isinstance(value, dict):
        return None
    input_tokens = value.get("prompt_tokens")
    output_tokens = value.get("completion_tokens")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in (input_tokens, output_tokens)
    ):
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _usage_tuple(value: dict[str, int] | None) -> tuple[int, int] | None:
    """Convert an artifact usage record to the ledger's tuple representation."""
    return None if value is None else (value["input_tokens"], value["output_tokens"])


def _usage_record(value: dict[str, int] | None) -> dict[str, Any]:
    """Retain whether usage was confirmed or remains unknown."""
    return {"status": "unknown"} if value is None else {**value, "status": "known"}


def _load_mapping(value: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    """Load a manifest mapping from memory or a UTF-8 JSON file."""
    if isinstance(value, Path):
        loaded = json.loads(value.read_text(encoding="utf-8"))
    else:
        loaded = value
    if not isinstance(loaded, Mapping):
        raise TypeError("MS3 diagnosis manifest must be a JSON object")
    return loaded


def _write_json(path: Path, value: Any) -> None:
    """Write one JSON artifact through a temporary file to avoid partial snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_bytes(path: Path, value: bytes) -> None:
    """Persist a non-successful raw HTTP response without text normalization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)
