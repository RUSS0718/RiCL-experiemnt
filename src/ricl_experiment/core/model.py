"""OpenAI-compatible HTTPX transport with auditable bounded retries."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import RunConfig, Sample, fingerprint
from .ledger import Ledger


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
    request_body = {
        "model": model.model,
        "messages": [{"role": "user", "content": sample.prompt}],
        **model.parameters,
    }
    request_fingerprint = fingerprint(request_body)
    ledger.ensure_call(call_id, sample.sample_id, request_fingerprint)
    previous = ledger.result(call_id)
    if previous is not None:
        return str(previous["response_ref"])
    ledger.recover_interrupted_attempts(call_id)
    recovered = _recover_saved_responses(ledger, run_dir, call_id)
    if recovered is not None:
        return recovered
    if ledger.call_status(call_id) == "failed":
        raise ModelDispatchError("logical call has a permanent failed attempt; start a new run")
    reservation = int(model.parameters.get("max_tokens", 0))
    key = os.environ.get(model.api_key_env) if model.api_key_env else None
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    endpoint = f"{model.base_url}/chat/completions"
    last_error = "no attempt was made"

    for _ in range(config.budget.max_attempts_per_call):
        try:
            attempt_id = ledger.start_attempt(
                call_id,
                request_fingerprint,
                model.parameters,
                reservation,
                config.budget.max_attempts_per_call,
            )
        except ValueError as error:
            raise ModelDispatchError(str(error)) from error
        started = time.monotonic()
        try:
            with httpx.Client(timeout=model.timeout_seconds, trust_env=False) as client:
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


def _usage(payload: dict[str, Any]) -> tuple[int, int] | None:
    """Extract validated OpenAI usage values when both token counts are present."""
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
    return input_tokens, output_tokens


def _save_response(run_dir: Path, attempt_id: str, content: bytes) -> str:
    """Persist response bytes under an attempt-specific, relative artifact reference."""
    relative = Path("responses") / f"{attempt_id}.json"
    path = run_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    os.replace(temporary_path, path)
    return relative.as_posix()

