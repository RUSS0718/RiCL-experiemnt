"""Blinded MS2 LLM-Rater and evidence scoring orchestration."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import RunConfig, fingerprint
from ..core.ledger import Ledger
from ..core.model import build_request_body, dispatch_request, read_response
from ..evaluation.evidence import (
    ClaimRecord,
    score_verified_progress,
    summarize_costs,
)

RATER_TEMPLATE_VERSION = "ms2-llm-rater-v1"
EVIDENCE_TEMPLATE_VERSION = "ms2-evidence-extraction-v1"
_SENSITIVE_KEYS = {
    "reference_answer",
    "reference_solution",
    "expert_label",
    "score",
    "sibling_rollout",
    "sibling_rollouts",
    "rollout",
    "rollouts",
}


def _validate_visible_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _safe_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    selected = dict(parameters or {})
    if _SENSITIVE_KEYS & selected.keys() or {"model", "messages"} & selected.keys():
        raise ValueError("scoring parameters cannot contain hidden or serialized request fields")
    return selected


def _build_request(
    model: str,
    problem: str,
    checkpoint: str,
    instruction: str,
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _validate_visible_text("problem", problem)
    _validate_visible_text("checkpoint", checkpoint)
    body = build_request_body(
        _validate_visible_text("model", model),
        "rater",
        {"problem": problem, "candidates": [checkpoint]},
        _safe_parameters(parameters),
    )
    body["messages"][0]["content"] += f"\n\n{instruction}"
    assert_no_blind_leakage(body)
    return body


def build_blinded_rater_request(
    model: str,
    problem: str,
    checkpoint: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact request visible to the isolated LLM-Rater role."""
    return _build_request(
        model,
        problem,
        checkpoint,
        (
            "Rater instructions:\n"
            "Assign one ordinal rating from 0 to 3 to the visible route. "
            "Return JSON with keys rating and rationale."
        ),
        parameters,
    )


def build_blinded_evidence_request(
    model: str,
    problem: str,
    checkpoint: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact request visible to the independent claim extractor."""
    return _build_request(
        model,
        problem,
        checkpoint,
        (
            "Evidence instructions:\n"
            "List each non-trivial mathematical claim visible in the checkpoint, its "
            "conditions, supporting steps, target relation, and source locator. "
            "Return JSON with key claims."
        ),
        parameters,
    )


def build_blinded_scoring_requests(
    model: str,
    problem: str,
    checkpoint: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "llm_rater": build_blinded_rater_request(model, problem, checkpoint, parameters),
        "evidence": build_blinded_evidence_request(model, problem, checkpoint, parameters),
    }


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            key for item in value.values() for key in _walk_keys(item)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {key for item in value for key in _walk_keys(item)}
    return set()


def assert_no_blind_leakage(
    request: Mapping[str, Any], *, forbidden_values: Sequence[str] = ()
) -> None:
    """Fail closed if a serialized role request contains restricted fields or test sentinels."""
    keys = {key.casefold() for key in _walk_keys(request)}
    sensitive = {key.casefold() for key in _SENSITIVE_KEYS}
    if keys & sensitive:
        raise ValueError(f"blinded request contains restricted field: {min(keys & sensitive)}")
    encoded = json.dumps(request, ensure_ascii=False, sort_keys=True)
    for value in forbidden_values:
        if value and value in encoded:
            raise ValueError("blinded request contains a forbidden value")


def _response_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("model response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str):
        raise TypeError("model response has no text content")
    return content


def _parse_json_object(content: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, Mapping) else None


def parse_llm_rater_response(content: str) -> dict[str, Any]:
    """Parse a controlled rater response while retaining raw text and parse status."""
    payload = _parse_json_object(content)
    if payload is None:
        return {"score": None, "rationale": "", "parse_status": "invalid", "raw_response": content}
    value = payload.get("rating", payload.get("score"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 3:
        return {
            "score": None,
            "rationale": str(payload.get("rationale", "")),
            "parse_status": "invalid",
            "raw_response": content,
        }
    return {
        "score": value,
        "rationale": str(payload.get("rationale", "")),
        "parse_status": "valid",
        "raw_response": content,
    }


def parse_claim_extraction_response(
    content: str, *, sample_id: str, route_id: str, problem: str, checkpoint: str
) -> dict[str, Any]:
    """Parse extracted claims as candidates; extraction never supplies a final verdict."""
    payload = _parse_json_object(content)
    if payload is None or not isinstance(payload.get("claims", []), list):
        return {"parse_status": "invalid", "claims": [], "raw_response": content}
    claims: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["claims"]):
        if not isinstance(raw, Mapping) or not str(raw.get("assertion", "")).strip():
            continue
        claim = ClaimRecord.from_mapping(
            {
                "claim_id": str(raw.get("claim_id", f"extracted-{index}")),
                "sample_id": sample_id,
                "route_id": route_id,
                "problem": problem,
                "checkpoint": checkpoint,
                "assertion": raw["assertion"],
                "conditions": raw.get("conditions", []),
                "target_relation": raw.get("target_relation", ""),
                "source_locator": raw.get("source_locator", ""),
                "milestones": raw.get("milestones", [1]),
                "provenance": "Unknown",
                "verdict": "unknown",
                "conditions_sufficient": None,
                "supporting_steps": raw.get("supporting_steps", []),
            }
        )
        claims.append(claim.to_dict())
    return {"parse_status": "valid", "claims": claims, "raw_response": content}


def _usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return {
            "status": "unknown",
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
        }
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    known = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (input_tokens, output_tokens))
    return {
        "status": "known" if known else "unknown",
        "input_tokens": input_tokens if known else None,
        "output_tokens": output_tokens if known else None,
        "reasoning_tokens": reasoning if isinstance(reasoning, int) and reasoning >= 0 else None,
    }


def _logical_call_id(run_id: str, sample_id: str, route_id: str, stage: str) -> str:
    return fingerprint({"run_id": run_id, "sample_id": sample_id, "route_id": route_id, "stage": stage})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _artifact_path(run_dir: Path, sample_id: str, route_id: str) -> Path:
    digest = hashlib.sha256(f"{sample_id}:{route_id}".encode()).hexdigest()[:24]
    return run_dir / "scoring" / f"{digest}.json"


def freeze_scoring_artifact(
    *,
    sample_id: str,
    route_id: str,
    problem: str,
    checkpoint: str,
    llm_rater: Mapping[str, Any],
    claim_extraction: Mapping[str, Any],
    verified_claims: Sequence[ClaimRecord | Mapping[str, Any]] = (),
    costs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create the persisted pre-rollout score artifact with no future outcomes attached."""
    progress = score_verified_progress(
        verified_claims,
        sample_id=sample_id,
        route_id=route_id,
    )
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "route_id": route_id,
        "problem": problem,
        "checkpoint": checkpoint,
        "templates": {
            "llm_rater": RATER_TEMPLATE_VERSION,
            "evidence": EVIDENCE_TEMPLATE_VERSION,
        },
        "llm_rater": dict(llm_rater),
        "claim_extraction": dict(claim_extraction),
        "verified_progress": progress,
        "cost": summarize_costs(costs),
        "associated_outcomes": [],
        "frozen_before_rollouts": True,
    }


def score_ms2_checkpoint(
    config: RunConfig,
    ledger: Ledger,
    run_dir: Path,
    *,
    run_id: str,
    sample_id: str,
    route_id: str,
    problem: str,
    checkpoint: str,
    verified_claims: Sequence[ClaimRecord | Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run both scoring roles through the audited transport using only controlled visible input."""
    parameters = dict(config.model.parameters)
    requests = build_blinded_scoring_requests(config.model.model, problem, checkpoint, parameters)
    calls = []
    parsed_rater = {"score": None, "rationale": "", "parse_status": "missing", "raw_response": ""}
    extraction = {"parse_status": "missing", "claims": [], "raw_response": ""}

    rater_id = _logical_call_id(run_id, sample_id, route_id, "llm-rater")
    rater_ref = dispatch_request(
        config,
        ledger,
        run_dir,
        rater_id,
        sample_id,
        requests["llm_rater"],
        effective_parameters=parameters,
    )
    rater_payload = read_response(run_dir, rater_ref)
    parsed_rater = parse_llm_rater_response(_response_text(rater_payload))
    calls.append(
        {
            "call_id": rater_id,
            "sample_id": sample_id,
            "route_id": route_id,
            "stage": "llm-rater",
            "usage": _usage(rater_payload),
        }
    )

    evidence_id = _logical_call_id(run_id, sample_id, route_id, "evidence")
    evidence_ref = dispatch_request(
        config,
        ledger,
        run_dir,
        evidence_id,
        sample_id,
        requests["evidence"],
        effective_parameters=parameters,
    )
    evidence_payload = read_response(run_dir, evidence_ref)
    extraction = parse_claim_extraction_response(
        _response_text(evidence_payload),
        sample_id=sample_id,
        route_id=route_id,
        problem=problem,
        checkpoint=checkpoint,
    )
    calls.append(
        {
            "call_id": evidence_id,
            "sample_id": sample_id,
            "route_id": route_id,
            "stage": "evidence",
            "usage": _usage(evidence_payload),
        }
    )

    artifact = freeze_scoring_artifact(
        sample_id=sample_id,
        route_id=route_id,
        problem=problem,
        checkpoint=checkpoint,
        llm_rater=parsed_rater,
        claim_extraction=extraction,
        verified_claims=verified_claims,
        costs=calls,
    )
    artifact["artifact_ref"] = str(_artifact_path(run_dir, sample_id, route_id).relative_to(run_dir))
    _write_json(_artifact_path(run_dir, sample_id, route_id), artifact)
    return artifact


# Stable public aliases for orchestration callers.
build_rater_request = build_blinded_rater_request
build_evidence_request = build_blinded_evidence_request
parse_rater_response = parse_llm_rater_response
score_checkpoint = score_ms2_checkpoint
