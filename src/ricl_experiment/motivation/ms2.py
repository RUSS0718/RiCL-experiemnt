"""MS2 Checkpoint creation, independent continuations, and trace export."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config import RunConfig, fingerprint, load_config, validate_api_key
from ..core.ledger import Ledger
from ..core.model import ModelDispatchError, dispatch_call, read_response
from ..evaluation.answer import classify_answer, extract_final_answer
from ..runner import _experiment_commit_sha, _run_lock
from .ms2_scoring import (
    EVIDENCE_REVIEW_TEMPLATE_VERSION,
    EVIDENCE_TEMPLATE_VERSION,
    RATER_TEMPLATE_VERSION,
    score_ms2_checkpoint,
)
from .schemas import (
    MS2AnswerKeyRow,
    MS2CandidateTrace,
    MS2Manifest,
    MS2Trace,
    base_trace,
    empty_rollout,
)

CONTINUATION_TEMPLATE_VERSION = "ms2-neutral-continuation-v1"
CONTINUATION_INSTRUCTIONS = (
    "Inspect the visible work, continue or correct it as needed, and explicitly state a final answer."
)


class MS2RunError(RuntimeError):
    """Raised when a frozen MS2 run cannot safely be created or resumed."""


def start_ms2_run(
    phase: str,
    config_path: Path,
    manifest_path: Path,
    answer_key_path: Path,
    artifacts_root: Path,
) -> tuple[str, str]:
    """Create a run and dispatch three candidates plus two rollouts per usable Checkpoint."""
    config, manifest, answer_bytes, answers = _inputs(config_path, manifest_path, answer_key_path)
    if phase != manifest.phase:
        raise ValueError("--phase must match the frozen MS2 manifest phase")
    run_id = uuid.uuid4().hex
    run_dir = _run_dir(artifacts_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    restricted = run_dir / "restricted"
    restricted.mkdir()
    (restricted / "answer-key.json").write_bytes(answer_bytes)
    metadata = _metadata(run_id, config, manifest, answer_bytes)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "config.json", config.redacted())
    _write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
    with _run_lock(run_dir):
        return _execute(run_id, run_dir, config, manifest, answers)


def resume_ms2_run(
    run_id: str,
    config_path: Path,
    manifest_path: Path,
    answer_key_path: Path,
    artifacts_root: Path,
) -> tuple[str, str]:
    """Resume stable logical calls only after validating every frozen input."""
    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in run_id
    ):
        raise ValueError("run_id contains invalid path characters")
    config, manifest, answer_bytes, answers = _inputs(
        config_path, manifest_path, answer_key_path
    )
    run_dir = _run_dir(artifacts_root, run_id)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    expected = _metadata(run_id, config, manifest, answer_bytes)
    for key in (
        "config_fingerprint",
        "manifest_fingerprint",
        "answer_key_fingerprint",
        "prompt_fingerprint",
        "continuation_template_version",
        "scoring_template_versions",
    ):
        if metadata.get(key) != expected[key]:
            raise MS2RunError(f"resume rejected: {key} does not match the original run")
    if metadata.get("run_id") != run_id:
        raise MS2RunError("resume rejected: run identity does not match the requested run_id")
    if (run_dir / "restricted" / "answer-key.json").read_bytes() != answer_bytes:
        raise MS2RunError("resume rejected: restricted answer key changed")
    with _run_lock(run_dir):
        return _execute(run_id, run_dir, config, manifest, answers)


def _inputs(
    config_path: Path, manifest_path: Path, answer_key_path: Path
) -> tuple[RunConfig, MS2Manifest, bytes, dict[str, MS2AnswerKeyRow]]:
    config = load_config(config_path)
    validate_api_key(config)
    try:
        manifest = MS2Manifest.model_validate_json(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ValueError(f"invalid MS2 manifest: {error}") from error
    answer_bytes = Path(answer_key_path).read_bytes()
    try:
        payload = json.loads(answer_bytes)
        if not isinstance(payload, list):
            raise TypeError("answer key must be a JSON array")
        rows = [MS2AnswerKeyRow.model_validate(row) for row in payload]
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise ValueError(f"invalid MS2 answer key: {error}") from error
    answers = {row.sample_id: row for row in rows}
    sample_ids = [sample.sample_id for sample in manifest.samples]
    if len(answers) != len(rows) or set(answers) != set(sample_ids):
        raise ValueError("answer-key sample IDs must exactly match the MS2 manifest")
    return config, manifest, answer_bytes, answers


def _execute(
    run_id: str,
    run_dir: Path,
    config: RunConfig,
    manifest: MS2Manifest,
    answers: dict[str, MS2AnswerKeyRow],
) -> tuple[str, str]:
    """Reconcile generation and rollout calls against the shared SQLite ledger."""
    ledger = Ledger(run_dir / "run.sqlite3", config.budget.max_reserved_output_tokens)
    failed_calls = False
    exported_samples: list[dict[str, Any]] = []
    try:
        for sample in manifest.samples:
            candidates: list[dict[str, Any]] = []
            answer = answers[sample.sample_id]
            for index in range(3):
                identity = {"run_id": run_id, "problem_id": sample.sample_id, "candidate_index": index}
                candidate_id = fingerprint(identity)
                route_id = fingerprint({**identity, "kind": "route"})
                candidate = {
                    "candidate_index": index,
                    "candidate_id": candidate_id,
                    "route_id": route_id,
                    "problem_id": sample.sample_id,
                    "problem_fingerprint": hashlib.sha256(
                        sample.problem.encode("utf-8")
                    ).hexdigest(),
                }
                generation_id = _logical_call_id(run_id, candidate_id, "initial")
                generation_parameters = _parameters(config, config.ms2.initial_max_tokens)
                _seed(generation_parameters, config, manifest, generation_id)
                try:
                    response_ref = dispatch_call(
                        config,
                        ledger,
                        run_dir,
                        generation_id,
                        sample.sample_id,
                        "generation",
                        {"problem": sample.problem},
                        generation_parameters,
                    )
                    payload = read_response(run_dir, response_ref)
                    prefix = payload["choices"][0]["message"]["content"]
                    stop_reason = payload["choices"][0].get("finish_reason")
                except ModelDispatchError as error:
                    failed_calls = True
                    candidate.update(
                        _checkpoint(candidate_id, generation_id, "", None, None, None, False)
                    )
                    candidate["generation_status"] = "failed"
                    candidate["generation_failure"] = str(error)
                    candidate["initial_evaluation"] = _evaluation(
                        "no_evaluable_output",
                        "early_unknown",
                        reason="initial generation failed before a response was received",
                        reference={"answer_key_ref": "restricted/answer-key.json", "sample_id": sample.sample_id},
                    )
                    candidate["rollouts"] = [
                        empty_rollout(i, candidate["prefix_fingerprint"], "initial_generation_failed")
                        for i in range(2)
                    ]
                    candidate["segments"] = [
                        {
                            "kind": "checkpoint_prefix",
                            "status": "unavailable",
                            "start": 0,
                            "end": 0,
                            "text": "",
                            "failure": str(error),
                        },
                        *[
                            {
                                "kind": f"rollout_{i}",
                                "status": "not_dispatched",
                                "start": 0,
                                "end": 0,
                                "text": "",
                                "failure": "initial_generation_failed",
                            }
                            for i in range(2)
                        ],
                    ]
                    generation_audit = ledger.call_audit(generation_id)
                    candidate["source_attempt_id"] = generation_audit["owner_attempt_id"]
                    candidate["source_attempts"] = generation_audit["attempts"]
                    candidate["attempts"] = generation_audit["attempts"]
                    candidates.append(candidate)
                    continue

                candidate.update(
                    _checkpoint(candidate_id, generation_id, prefix, response_ref, payload, stop_reason, True)
                )
                decision = classify_answer(prefix, answer.reference_answer)
                category = {
                    "correct": "early_correct",
                    "incorrect": "early_incorrect",
                }.get(decision.status.value, "early_unknown")
                candidate["generation_status"] = "completed"
                generation_audit = ledger.call_audit(generation_id)
                candidate["source_attempt_id"] = generation_audit["owner_attempt_id"]
                candidate["source_attempts"] = generation_audit["attempts"]
                candidate["attempts"] = generation_audit["attempts"]
                candidate["initial_evaluation"] = _evaluation(
                    decision.status.value,
                    category,
                    extracted_answer=decision.extracted_answer,
                    rule_version=decision.rule_version,
                    reason=decision.reason,
                    reference={"answer_key_ref": "restricted/answer-key.json", "sample_id": sample.sample_id},
                )
                candidate["rollouts"] = []
                candidate["segments"] = [
                    {
                        "kind": "checkpoint_prefix",
                        "status": "completed",
                        "start": 0,
                        "end": len(prefix),
                        "text": prefix,
                    }
                ]
                if config.ms2.score_checkpoints and prefix.strip():
                    try:
                        score = score_ms2_checkpoint(
                            config, ledger, run_dir,
                            run_id=run_id,
                            sample_id=sample.sample_id,
                            route_id=route_id,
                            problem=sample.problem,
                            checkpoint=prefix,
                            review_extracted_claims=True,
                        )
                        candidate["scoring_artifact_ref"] = score["artifact_ref"]
                    except ModelDispatchError as error:
                        failed_calls = True
                        candidate["scoring_failure"] = str(error)
                for rollout_index in range(2):
                    rollout_id = _logical_call_id(
                        run_id, candidate["checkpoint_id"], f"rollout-{rollout_index}"
                    )
                    prefix_fingerprint = candidate["prefix_fingerprint"]
                    segment_start = len(prefix)
                    if not prefix.strip():
                        rollout = empty_rollout(rollout_index, prefix_fingerprint, "empty_prefix")
                        candidate["segments"].append(
                            {
                                "kind": f"rollout_{rollout_index}",
                                "status": "not_dispatched",
                                "start": segment_start,
                                "end": segment_start,
                                "text": "",
                            }
                        )
                    else:
                        parameters = _parameters(config, config.ms2.continuation_max_tokens)
                        _seed(parameters, config, manifest, rollout_id)
                        try:
                            rollout_ref = dispatch_call(
                                config,
                                ledger,
                                run_dir,
                                rollout_id,
                                sample.sample_id,
                                "continuation",
                                {
                                    "problem": sample.problem,
                                    "checkpoint": prefix,
                                    "instruction": CONTINUATION_INSTRUCTIONS,
                                },
                                parameters,
                            )
                            rollout_payload = read_response(run_dir, rollout_ref)
                            rollout_audit = ledger.call_audit(rollout_id)
                            text = rollout_payload["choices"][0]["message"]["content"]
                            rollout = _rollout(
                                rollout_index,
                                rollout_id,
                                prefix_fingerprint,
                                prefix,
                                rollout_ref,
                                text,
                                rollout_payload,
                                answer,
                                sample.sample_id,
                            )
                            start = len(prefix)
                            candidate["segments"].append(
                                {
                                    "kind": f"rollout_{rollout_index}",
                                    "status": "completed",
                                    "start": start,
                                    "end": start + len(text),
                                    "text": text,
                                    "response_ref": rollout_ref,
                                    "attempts": rollout_audit["attempts"],
                                }
                            )
                        except ModelDispatchError as error:
                            failed_calls = True
                            failed_audit = ledger.call_audit(rollout_id)
                            rollout = _failed_rollout(
                                rollout_index,
                                rollout_id,
                                prefix_fingerprint,
                                error,
                                failed_audit,
                                sample.sample_id,
                            )
                            candidate["segments"].append(
                                {
                                    "kind": f"rollout_{rollout_index}",
                                    "status": "failed",
                                    "start": segment_start,
                                    "end": segment_start,
                                    "text": "",
                                    "failure": str(error),
                                    "attempts": failed_audit["attempts"],
                                    "response_ref": failed_audit.get("response_ref"),
                                }
                            )
                    rollout["attempts"] = ledger.call_audit(rollout_id)["attempts"] if rollout["status"] == "completed" else rollout.get("attempts", [])
                    candidate["rollouts"].append(rollout)
                candidates.append(candidate)
            exported_samples.append(
                {
                    **sample.model_dump(mode="json", exclude_none=True),
                    "candidates": candidates,
                }
            )
        status = "incomplete" if failed_calls else "completed"
        for sample_trace in exported_samples:
            for candidate in sample_trace["candidates"]:
                MS2CandidateTrace.model_validate(candidate)
        ledger.close_run(status)
        trace = base_trace(run_id, manifest, fingerprint(manifest.model_dump(mode="json")))
        trace["samples"] = exported_samples
        trace["usage"] = ledger.usage_summary()
        trace = MS2Trace.model_validate(trace).model_dump(mode="json")
        _write_json(run_dir / "traces" / "ms2.json", trace)
    finally:
        ledger.close()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["status"] = status
    metadata["usage"] = trace["usage"]
    metadata["trace_ref"] = "traces/ms2.json"
    _write_json(run_dir / "metadata.json", metadata)
    return run_id, status


def _checkpoint(
    candidate_id: str,
    generation_id: str,
    prefix: str,
    response_ref: str | None,
    payload: dict[str, Any] | None,
    stop_reason: Any,
    received: bool,
) -> dict[str, Any]:
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    reason = stop_reason if isinstance(stop_reason, str) and stop_reason else None
    budget_hit = True if reason == "length" else False if reason == "stop" else None
    return {
        "checkpoint_id": fingerprint({"candidate_id": candidate_id, "prefix_fingerprint": digest}),
        "prefix_text": prefix,
        "prefix_fingerprint": digest,
        "prefix_status": "available" if received and prefix.strip() else "empty" if received else "unavailable",
        "source_logical_call_id": generation_id,
        "source_response_ref": response_ref,
        "stop_reason": reason,
        "budget_hit": budget_hit,
        "budget_hit_basis": "configured_max_tokens_limit" if reason == "length" else None,
        "truncation_status": "truncated" if reason == "length" else "not_truncated" if reason == "stop" else "unknown",
        "last_step_status": (
            "complete" if reason == "stop" else "incomplete" if reason == "length" else "unknown"
        ),
        "visible_final_answer": extract_final_answer(prefix) is not None,
        "usage": _usage(payload) if payload is not None else None,
    }


def _rollout(
    index: int,
    call_id: str,
    prefix_fingerprint: str,
    prefix: str,
    response_ref: str,
    text: str,
    payload: dict[str, Any],
    answer: MS2AnswerKeyRow,
    sample_id: str,
) -> dict[str, Any]:
    decision = classify_answer(text, answer.reference_answer)
    return {
        "rollout_index": index,
        "status": "completed",
        "logical_call_id": call_id,
        "checkpoint_prefix_fingerprint": prefix_fingerprint,
        "response_ref": response_ref,
        "text": text,
        "usage": _usage(payload),
        "evaluation_reference": {
            "answer_key_ref": "restricted/answer-key.json",
            "sample_id": sample_id,
            "method": decision.rule_version,
        },
        "evaluation": _evaluation(
            decision.status.value,
            extracted_answer=decision.extracted_answer,
            normalized_answer=decision.normalized_answer,
            rule_version=decision.rule_version,
            reason=decision.reason,
        ),
    }


def _failed_rollout(
    index: int,
    call_id: str,
    prefix_fingerprint: str,
    error: ModelDispatchError,
    audit: dict[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    attempts = audit["attempts"]
    return {
        "rollout_index": index,
        "status": "failed",
        "logical_call_id": call_id,
        "checkpoint_prefix_fingerprint": prefix_fingerprint,
        "response_ref": audit.get("response_ref"),
        "failure": str(error),
        "attempts": attempts,
        "usage": _attempt_usage(attempts),
        "evaluation_reference": {
            "answer_key_ref": "restricted/answer-key.json",
            "sample_id": sample_id,
        },
    }


def _evaluation(
    status: str,
    category: str | None = None,
    *,
    extracted_answer: str | None = None,
    normalized_answer: str | None = None,
    rule_version: str | None = None,
    reason: str | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if category is not None:
        result["analysis_category"] = category
    if extracted_answer is not None:
        result["extracted_answer"] = extracted_answer
    if normalized_answer is not None:
        result["normalized_answer"] = normalized_answer
    if rule_version is not None:
        result["rule_version"] = rule_version
    if reason is not None:
        result["reason"] = reason
    if reference is not None:
        result["reference"] = reference
    return result


def _parameters(config: RunConfig, max_tokens: int) -> dict[str, Any]:
    parameters = dict(config.model.parameters)
    parameters["max_tokens"] = max_tokens
    return parameters


def _seed(
    parameters: dict[str, Any], config: RunConfig, manifest: MS2Manifest, call_id: str
) -> None:
    if "seed" not in config.model.capabilities.supported_parameters:
        return
    sampling = manifest.model_extra.get("sampling", {}) if manifest.model_extra else {}
    seed = sampling.get("seed") if isinstance(sampling, dict) else None
    if type(seed) is int and seed >= 0:
        digest = hashlib.sha256(f"{seed}:{call_id}".encode()).digest()
        parameters["seed"] = int.from_bytes(digest[:4], "big")


def _logical_call_id(run_id: str, identity: str, stage: str) -> str:
    return fingerprint({"run_id": run_id, "identity": identity, "stage": stage})


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {
            "status": "unknown",
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
        }
    input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    details = usage.get("completion_tokens_details")
    reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if reasoning_tokens is not None and (
        type(reasoning_tokens) is not int or reasoning_tokens < 0
    ):
        reasoning_tokens = None
    if (
        type(input_tokens) is int
        and input_tokens >= 0
        and type(output_tokens) is int
        and output_tokens >= 0
    ):
        return {
            "status": "known",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
    return {
        "status": "unknown",
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
    }


def _attempt_usage(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: attempt[key]
            for key in (
                "attempt_id",
                "attempt_number",
                "status",
                "usage_status",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "reserved_output_tokens",
                "response_ref",
                "error",
            )
        }
        for attempt in attempts
    ]


def _metadata(
    run_id: str, config: RunConfig, manifest: MS2Manifest, answer_bytes: bytes
) -> dict[str, Any]:
    manifest_data = manifest.model_dump(mode="json")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "study": "ms2",
        "phase": manifest.phase,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit_sha": _experiment_commit_sha(),
        "status": "created",
        "config_fingerprint": fingerprint(config.redacted()),
        "manifest_fingerprint": fingerprint(manifest_data),
        "answer_key_fingerprint": hashlib.sha256(answer_bytes).hexdigest(),
        "prompt_fingerprint": fingerprint([sample.problem for sample in manifest.samples]),
        "experiment_id": manifest.manifest_id,
        "model": config.model.model,
        "endpoint": config.model.base_url,
        "effective_parameters": config.model.parameters,
        "ms2_parameters": config.ms2.model_dump(mode="json"),
        "continuation_template_version": CONTINUATION_TEMPLATE_VERSION,
        "scoring_template_versions": (
            [RATER_TEMPLATE_VERSION, EVIDENCE_TEMPLATE_VERSION, EVIDENCE_REVIEW_TEMPLATE_VERSION]
            if config.ms2.score_checkpoints else None
        ),
        "redacted_config": config.redacted(),
    }


def _run_dir(root: Path, run_id: str) -> Path:
    return root.resolve() / "motivation" / run_id


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
