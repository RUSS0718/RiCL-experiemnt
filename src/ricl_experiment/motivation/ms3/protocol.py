"""MS3 diagnosis request protocol, validation, and transport boundary."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ...config import RunConfig
from ...core.model import build_request_body

METHODS: tuple[dict[str, str], ...] = (
    {
        "id": "direct_judge",
        "instruction": (
            "Judge the complete trace directly. Decide whether any step is erroneous and, "
            "if so, give the first erroneous zero-based step."
        ),
    },
    {
        "id": "step_by_step_critic",
        "instruction": (
            "Critique the trace step by step in order. Decide whether any step is erroneous "
            "and identify the first erroneous zero-based step."
        ),
    },
    {
        "id": "reflection",
        "instruction": (
            "Reflect on possible failure causes in the complete trace before deciding whether "
            "an error exists and locating its first erroneous zero-based step."
        ),
    },
)
METHOD_IDS = tuple(method["id"] for method in METHODS)
TEMPLATE_VERSION = "ms3-diagnosis-v1"


@dataclass(frozen=True)
class DiagnosisSample:
    """Public ProcessBench fields that a diagnosis model may receive."""

    sample_id: str
    problem: str
    steps: tuple[str, ...]
    trajectory_id: str
    question_group_id: str
    source_record_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DiagnosisSample:
        """Validate one public trace while ignoring restricted evaluation fields."""
        sample_id = _required_text(value, "sample_id")
        problem = _required_text(value, "problem")
        steps = _steps(value.get("steps"))
        if len(steps) < 2 or any(not isinstance(step, str) or not step.strip() for step in steps):
            raise ValueError(f"{sample_id}: steps must contain at least two non-empty strings")
        trajectory_id = _required_text(value, "trajectory_id")
        question_group_id = _required_text(value, "question_group_id")
        return cls(
            sample_id=sample_id,
            problem=problem,
            steps=tuple(steps),
            trajectory_id=trajectory_id,
            question_group_id=question_group_id,
            source_record_id=_optional_text(value.get("source_record_id")),
        )

    def public_record(self) -> dict[str, Any]:
        """Return stable identity and trace fields without restricted labels or answers."""
        result = {
            "sample_id": self.sample_id,
            "problem": self.problem,
            "steps": list(self.steps),
            "trajectory_id": self.trajectory_id,
            "question_group_id": self.question_group_id,
        }
        if self.source_record_id is not None:
            result["source_record_id"] = self.source_record_id
        return result


@dataclass(frozen=True)
class DiagnosisReply:
    """Normalized successful chat response returned by a transport seam."""

    content: str
    raw_response: Mapping[str, Any]
    usage: dict[str, int] | None
    elapsed_ms: int


class DiagnosisRuntimeError(RuntimeError):
    """A local transport failure that was not an HTTP API response."""

    def __init__(self, message: str, elapsed_ms: int | None = None):
        """Retain elapsed time for the failure audit row."""
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


class DiagnosisAPIError(RuntimeError):
    """An endpoint or transport failure recorded separately from invalid output."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: bytes | None = None,
        usage: dict[str, int] | None = None,
        elapsed_ms: int | None = None,
        retryable: bool = True,
    ):
        """Retain bounded retry, usage, and raw-body details for the audit ledger."""
        super().__init__(message)
        self.raw_response = raw_response
        self.usage = usage
        self.elapsed_ms = elapsed_ms
        self.retryable = retryable


class DiagnosisTransport(Protocol):
    """Transport seam used by tests and the real OpenAI-compatible adapter."""

    def send(self, request_body: dict[str, Any]) -> DiagnosisReply:
        """Send one already-isolated request and return its raw assistant response."""


class HttpxDiagnosisTransport:
    """Small OpenAI-compatible transport used by the MS3 CLI."""

    def __init__(self, config: RunConfig):
        """Keep endpoint settings local and never persist the resolved API key."""
        self._config = config

    def send(self, request_body: dict[str, Any]) -> DiagnosisReply:
        """Send one request and reject non-chat or non-success responses."""
        key = os.environ.get(self._config.model.api_key_env) if self._config.model.api_key_env else None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        started = time.monotonic()
        try:
            with httpx.Client(timeout=self._config.model.timeout_seconds, trust_env=False) as client:
                response = client.post(
                    f"{self._config.model.base_url}/chat/completions",
                    json=request_body,
                    headers=headers,
                )
        except httpx.HTTPError as error:
            raise DiagnosisRuntimeError(
                type(error).__name__, elapsed_ms=int((time.monotonic() - started) * 1000)
            ) from error
        elapsed_ms = int((time.monotonic() - started) * 1000)
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as error:
            raise DiagnosisAPIError(
                f"response was not JSON: {type(error).__name__}",
                raw_response=response.content,
                elapsed_ms=elapsed_ms,
                retryable=response.status_code >= 500 or response.status_code == 429,
            ) from error
        if not response.is_success:
            raise DiagnosisAPIError(
                f"model endpoint returned HTTP {response.status_code}",
                raw_response=response.content,
                usage=_usage(payload.get("usage")) if isinstance(payload, dict) else None,
                elapsed_ms=elapsed_ms,
                retryable=response.status_code >= 500 or response.status_code == 429,
            )
        if not _is_chat_completion(payload):
            raise DiagnosisAPIError(
                "model response did not contain one assistant message",
                raw_response=response.content,
                usage=_usage(payload.get("usage")) if isinstance(payload, dict) else None,
                elapsed_ms=elapsed_ms,
                retryable=False,
            )
        return DiagnosisReply(
            content=payload["choices"][0]["message"]["content"],
            raw_response=payload,
            usage=_usage(payload.get("usage")),
            elapsed_ms=elapsed_ms,
        )


def build_diagnosis(
    method: Mapping[str, str],
    sample: Mapping[str, Any],
    model: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one blind, tool-disabled request from only the public trace fields."""
    if method.get("id") not in METHOD_IDS:
        raise ValueError(f"unknown MS3 diagnosis method: {method.get('id')}")
    tool_parameters = {"tools", "tool_choice", "parallel_tool_calls"} & parameters.keys()
    if tool_parameters:
        raise ValueError("MS3 diagnosis model parameters must not configure tools")
    steps = _steps(sample.get("steps"))
    if len(steps) < 2 or any(not isinstance(step, str) or not step.strip() for step in steps):
        raise ValueError("MS3 diagnosis requires at least two non-empty text steps")
    body = build_request_body(
        model,
        "diagnosis",
        {
            "problem": _required_text(sample, "problem"),
            "steps": tuple(f"Step {index}:\n{step}" for index, step in enumerate(steps)),
        },
        dict(parameters),
    )
    body["tool_choice"] = "none"
    contract = (
        "Return JSON only as an object with required fields has_error (true, false, or null) and "
        "first_error_step (a zero-based integer in range or null). You may also include "
        "error_type and explanation as strings or null, and confidence as a number from 0 to 1 "
        "for a submitted judgment or null when abstaining. Use true with a legal index or null, "
        "false with null, or null with null."
    )
    body["messages"][0]["content"] = (
        f"Method {method['id']}.\n{method['instruction']}\n\n"
        f"{contract}\n\n{body['messages'][0]['content']}"
    )
    return body


def parse_diagnosis(raw_output: str, step_count: int) -> tuple[str, dict[str, Any] | None, str | None]:
    """Parse and validate the frozen diagnosis contract without consulting ground truth."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return "parse_error", None, "empty model output"
    candidate = raw_output.strip()
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as error:
        return "parse_error", None, f"invalid JSON: {type(error).__name__}"
    if not isinstance(value, dict):
        return "invalid_output", None, "diagnosis must be a JSON object"
    required = {"has_error", "first_error_step"}
    allowed = required | {"error_type", "explanation", "confidence"}
    if not required.issubset(value) or set(value) - allowed:
        return "invalid_output", None, "diagnosis fields do not match the contract"
    has_error = value["has_error"]
    first_step = value["first_error_step"]
    if has_error is not None and not isinstance(has_error, bool):
        return "invalid_output", None, "has_error must be bool or null"
    if first_step is not None and (
        isinstance(first_step, bool) or not isinstance(first_step, int)
    ):
        return "invalid_output", None, "first_error_step must be int or null"
    if first_step is not None and not 0 <= first_step < step_count:
        return "invalid_output", None, "first_error_step is outside the trace"
    if (has_error is False or has_error is None) and first_step is not None:
        return "invalid_output", None, "first_error_step contradicts has_error"
    for field in ("error_type", "explanation"):
        if value.get(field) is not None and not isinstance(value[field], str):
            return "invalid_output", None, f"{field} must be string or null"
    confidence = value.get("confidence")
    if "confidence" in value and has_error is None and confidence is not None:
        return "invalid_output", None, "confidence must be null when abstaining"
    if "confidence" in value and has_error is not None and confidence is None:
        return "invalid_output", None, "confidence must be numeric for a judgment"
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        return "invalid_output", None, "confidence must be a number from 0 to 1 or null"
    diagnosis = {"has_error": has_error, "first_error_step": first_step}
    for field in ("error_type", "explanation"):
        if field in value:
            diagnosis[field] = value[field]
    if confidence is not None:
        diagnosis["confidence"] = float(confidence)
    return ("abstained" if has_error is None else "completed"), diagnosis, None




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


def _steps(value: Any) -> list[str]:
    """Convert supported trace encodings without trimming non-empty step text."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return value.splitlines()
    raise ValueError("steps must be a list of strings or newline-delimited text")


def _required_text(value: Mapping[str, Any], name: str) -> str:
    """Require a non-empty string field while retaining its original value."""
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _optional_text(value: Any) -> str | None:
    """Return a non-empty string or no value."""
    return value if isinstance(value, str) and value.strip() else None
