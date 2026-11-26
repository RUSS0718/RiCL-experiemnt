"""OpenAI-compatible HTTPX transport with auditable bounded retries."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from ..config import RunConfig, Sample, fingerprint
from .ledger import Ledger

ROLE_FIELD_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "generation": ("problem", "instruction"),
    "continuation": ("problem", "checkpoint", "instruction"),
    "rater": ("problem", "candidates"),
    "diagnosis": ("problem", "steps"),
    "repair": ("problem", "steps"),
    "oracle_repair": ("problem", "steps", "first_error_position"),
    "ms3_repair": ("problem", "steps", "repair_instruction"),
    "independent_evaluation": (
        "problem", "candidate_output", "reference_answer", "reference_solution"
    ),
    "blind_review": ("problem", "candidate_output", "reference_material"),
}


def build_request_body(
    model: str, role: str, visible_fields: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    """Serialize only allowlisted values into the actual OpenAI-compatible request shape."""
    if role not in ROLE_FIELD_ALLOWLIST:
        raise ValueError(f"unknown request role: {role}")
    if not isinstance(visible_fields.get("problem"), str) or not visible_fields["problem"].strip():
        raise ValueError(f"{role} request requires a non-empty problem")
    if {"model", "messages"} & parameters.keys():
        raise ValueError("model parameters cannot replace role-isolated request fields")
    selected = {
        name: visible_fields[name]
        for name in ROLE_FIELD_ALLOWLIST[role]
        if name in visible_fields and visible_fields[name] is not None
    }
    for name in ("steps", "candidates"):
        if name in selected and (
            not isinstance(selected[name], (list, tuple))
            or any(not isinstance(item, str) for item in selected[name])
        ):
            raise ValueError(f"{role} field {name} must contain only anonymous text")
    for name in (
        "checkpoint",
        "candidate_output",
        "reference_answer",
        "reference_solution",
        "reference_material",
        "repair_instruction",
        "instruction",
    ):
        if name in selected and not isinstance(selected[name], str):
            raise ValueError(f"{role} field {name} must be plain text")
    position = selected.get("first_error_position")
    if position is not None and (isinstance(position, bool) or not isinstance(position, int) or position < 0):
        raise ValueError("first_error_position must be a non-negative integer")
    if role == "generation":
        content = selected["problem"]
        if "instruction" in selected:
            content += f"\n\nInstruction:\n{selected['instruction']}"
    else:
        labels = {
            "problem": "Problem",
            "checkpoint": "Checkpoint",
            "candidates": "Anonymous candidates",
            "steps": "Trace steps",
            "first_error_position": "First-error position",
            "candidate_output": "Candidate output",
            "reference_answer": "Reference answer",
            "reference_solution": "Reference solution",
            "reference_material": "Reference material",
            "repair_instruction": "Repair instruction",
            "instruction": "Instruction",
        }
        sections = []
        for name, value in selected.items():
            if isinstance(value, (list, tuple)):
                rendered = "\n".join(value)
            else:
                rendered = str(value)
            sections.append(f"{labels[name]}:\n{rendered}")
        content = "\n\n".join(sections)
    return {
        **parameters,
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }


class ModelDispatchError(RuntimeError):
    """Raised when a call cannot complete within configured attempt or budget limits."""


def dispatch_sample(
    config: RunConfig,
    ledger: Ledger,
    run_dir: Path,
    call_id: str,
    sample: Sample,
) -> str:
    """Send one user-only request, recording attempts and the first durable response."""
    model = config.model
    request_body = build_request_body(
        model.model, "generation", {"problem": sample.prompt}, model.parameters
    )
    return dispatch_request(
        config,
        ledger,
        run_dir,
        call_id,
        sample.sample_id,
        request_body,
    )


def dispatch_call(
    config: RunConfig,
    ledger: Ledger,
    run_dir: Path,
    call_id: str,
    sample_id: str,
    role: str,
    visible_fields: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    *,
    redispatch_failed: bool = False,
) -> str:
    """Build and send one role-isolated request with stage-specific parameters."""
    model = config.model
    effective_parameters = dict(model.parameters if parameters is None else parameters)
    unsupported = effective_parameters.keys() - set(model.capabilities.supported_parameters)
    missing = set(model.capabilities.required_parameters) - effective_parameters.keys()
    if unsupported or missing:
        raise ValueError("effective model parameters do not match configured capabilities")
    max_tokens = effective_parameters.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("effective max_tokens must be a positive integer")
    request_body = build_request_body(model.model, role, visible_fields, effective_parameters)
    return dispatch_request(
        config,
        ledger,
        run_dir,
        call_id,
        sample_id,
        request_body,
        effective_parameters=effective_parameters,
        redispatch_failed=redispatch_failed,
    )


def dispatch_request(
    config: RunConfig,
    ledger: Ledger,
    run_dir: Path,
    call_id: str,
    sample_id: str,
    request_body: dict[str, Any],
    *,
    effective_parameters: dict[str, Any] | None = None,
    redispatch_failed: bool = False,
) -> str:
    """Send one isolated request through the shared audited transport."""
    model = config.model
    effective_parameters = dict(model.parameters if effective_parameters is None else effective_parameters)
    request_fingerprint = fingerprint(request_body)
    ledger.ensure_call(call_id, sample_id, request_fingerprint, request_body=request_body)
    previous = ledger.result(call_id)
    if previous is not None:
        return str(previous["response_ref"])
    ledger.recover_interrupted_attempts(call_id)
    recovered = _recover_saved_responses(ledger, run_dir, call_id)
    if recovered is not None:
        return recovered
    if ledger.call_status(call_id) == "failed":
        if not redispatch_failed:
            raise ModelDispatchError("logical call has a permanent failed attempt; start a new run")
        ledger.authorize_redispatch(call_id, request_fingerprint)
    reservation = int(effective_parameters.get("max_tokens", 0))
    key = os.environ.get(model.api_key_env) if model.api_key_env else None
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    endpoint = f"{model.base_url}/chat/completions"
    last_error = "no attempt was made"

    for _ in range(config.budget.max_attempts_per_call):
        try:
            attempt_id = ledger.start_attempt(
                call_id,
                request_fingerprint,
                effective_parameters,
                reservation,
                config.budget.max_attempts_per_call,
            )
        except ValueError as error:
            raise ModelDispatchError(str(error)) from error
        started = time.monotonic()
        try:
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            with httpx.Client(
                timeout=model.timeout_seconds, trust_env=False, proxy=proxy
            ) as client:
                response = client.post(endpoint, json=request_body, headers=headers)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            response_ref = _save_response(run_dir, attempt_id, response.content)
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError):
                payload = {}
            usage = _usage(payload) if isinstance(payload, dict) else None
            if response.is_success:
                if not _is_chat_completion(payload):
                    ledger.record_failure(
                        attempt_id,
                        "failed",
                        f"invalid model response shape; elapsed_ms={elapsed_ms}; response_ref={response_ref}",
                        usage=usage,
                        response_ref=response_ref,
                    )
                    raise ModelDispatchError("model returned an incomplete chat completion")
                ledger.record_response(call_id, attempt_id, response_ref, usage)
                return response_ref
            code = response.status_code
            transient = code >= 500 or code == 429
            status = "unknown" if code >= 500 else "retryable" if code == 429 else "failed"
            ledger.record_failure(
                attempt_id,
                status,
                f"HTTP {code}; elapsed_ms={elapsed_ms}; response_ref={response_ref}",
                usage=usage,
                response_ref=response_ref,
            )
            last_error = f"model endpoint returned HTTP {code}"
            if not transient:
                break
        except httpx.TimeoutException as error:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            ledger.record_failure(
                attempt_id,
                "unknown",
                f"{type(error).__name__}; elapsed_ms={elapsed_ms}",
                usage=None,
            )
            last_error = f"model request timed out after {elapsed_ms} ms"
        except httpx.TransportError as error:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            ledger.record_failure(
                attempt_id,
                "unknown",
                f"{type(error).__name__}; elapsed_ms={elapsed_ms}",
                usage=None,
            )
            last_error = f"model transport did not confirm a response ({type(error).__name__})"
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            ledger.record_failure(
                attempt_id,
                "failed",
                f"invalid model response: {type(error).__name__}; elapsed_ms={elapsed_ms}",
                usage=None,
            )
            last_error = f"model returned an invalid response ({type(error).__name__})"
            break

    raise ModelDispatchError(
        f"{last_error}; retry again with the same run_id if attempts remain"
    )


def usage_out_of_bounds(attempt: Mapping[str, Any]) -> bool:
    """Detect a reported completion count or reasoning subtotal beyond its bound."""
    output = attempt["output_tokens"]
    reasoning = attempt["reasoning_tokens"]
    return output is not None and (
        output > attempt["reserved_output_tokens"]
        or (reasoning is not None and reasoning > output)
    )


def read_response(run_dir: Path, response_ref: str) -> dict[str, Any]:
    """Read and validate a persisted chat completion without making another request."""
    response_path = run_dir / response_ref
    if response_path.is_symlink() or not response_path.is_file():
        raise ValueError("response artifact is not a regular file")
    payload = json.loads(response_path.read_bytes())
    if not isinstance(payload, dict) or not _is_chat_completion(payload):
        raise ValueError("response does not match chat completion shape")
    return payload


def _recover_saved_responses(ledger: Ledger, run_dir: Path, call_id: str) -> str | None:
    """Claim a complete response only from its unresolved attempt's exact artifact path."""
    response_dir = run_dir / "responses"
    for row in ledger.unrecorded_attempts(call_id):
        attempt_id = row["attempt_id"]
        if not isinstance(attempt_id, str) or not re.fullmatch(r"[a-f0-9]{32}", attempt_id):
            continue
        response_ref = f"responses/{attempt_id}.json"
        response_path = run_dir / response_ref
        if not response_path.exists() and not response_path.is_symlink():
            continue
        try:
            if response_dir.is_symlink() or response_path.is_symlink() or not response_path.is_file():
                raise ValueError("response artifact is not a regular attempt file")
            payload = json.loads(response_path.read_bytes())
            if not isinstance(payload, dict) or not _is_chat_completion(payload):
                raise ValueError("response does not match chat completion shape")
        except (OSError, ValueError) as error:
            ledger.record_failure(
                attempt_id,
                "unknown",
                f"invalid persisted response: {type(error).__name__}",
                usage=None,
                response_ref=response_ref,
            )
            continue
        ledger.record_response(call_id, attempt_id, response_ref, _usage(payload))
        return response_ref
    return None


def _is_chat_completion(payload: dict[str, Any]) -> bool:
    """Recognize the non-streaming assistant message shape this runner consumes."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    return isinstance(message, dict) and isinstance(message.get("content"), str)


def _usage(payload: dict[str, Any]) -> tuple[int, int, int | None] | None:
    """Extract validated input, output, and optional reasoning token counts."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if (
        isinstance(input_tokens, bool)
        or isinstance(output_tokens, bool)
        or not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or input_tokens < 0
        or output_tokens < 0
    ):
        return None
    details = usage.get("completion_tokens_details")
    reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if reasoning_tokens is not None and (
        isinstance(reasoning_tokens, bool)
        or not isinstance(reasoning_tokens, int)
        or reasoning_tokens < 0
    ):
        reasoning_tokens = None
    return input_tokens, output_tokens, reasoning_tokens


def _save_response(run_dir: Path, attempt_id: str, content: bytes) -> str:
    """Persist response bytes under an attempt-specific, relative artifact reference."""
    relative = Path("responses") / f"{attempt_id}.json"
    path = run_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    os.replace(temporary_path, path)
    return relative.as_posix()

