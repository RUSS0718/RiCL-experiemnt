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
EVIDENCE_TEMPLATE_VERSION = "ms2-evidence-extraction-v2"
EVIDENCE_REVIEW_TEMPLATE_VERSION = "ms2-evidence-review-v1"
JSON_EVIDENCE_TEMPLATE_VERSION = "ms2-evidence-extraction-json-v1"
JSON_EVIDENCE_REVIEW_TEMPLATE_VERSION = "ms2-evidence-review-json-v1"
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
    *,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Build the exact request visible to the independent claim extractor."""
    request = _build_request(
        model,
        problem,
        checkpoint,
        (
            "Evidence instructions:\n"
            "Return only a json object, for example: {\"claims\": []}. Each claim must include assertion "
            "(a non-trivial mathematical claim visible in the checkpoint), conditions "
            "(array of strings), supporting_steps (array of strings), target_relation "
            "(string), and source_locator (an exact quote from the checkpoint). "
            "Return an empty array only when there are no relevant non-trivial claims."
        ),
        parameters,
    )
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    return request


def build_blinded_scoring_requests(
    model: str,
    problem: str,
    checkpoint: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    json_evidence: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        "llm_rater": build_blinded_rater_request(model, problem, checkpoint, parameters),
        "evidence": build_blinded_evidence_request(
            model, problem, checkpoint, parameters, json_mode=json_evidence
        ),
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
    if payload is None or not isinstance(payload.get("claims"), list):
        return {"parse_status": "invalid", "claims": [], "raw_response": content}
    claims: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["claims"]):
        if (
            not isinstance(raw, Mapping)
            or not isinstance(raw.get("assertion"), str)
            or not raw["assertion"].strip()
        ):
            return {"parse_status": "invalid", "claims": [], "raw_response": content}
        claim = ClaimRecord.from_mapping(
            {
                "claim_id": f"extracted-{index}",
                "sample_id": sample_id,
                "route_id": route_id,
                "problem": problem,
                "checkpoint": checkpoint,
                "assertion": raw["assertion"],
                "conditions": raw.get("conditions", []),
                "target_relation": raw.get("target_relation", ""),
                "source_locator": raw.get("source_locator", ""),
                "milestones": [1],
                "provenance": "Unknown",
                "verdict": "unknown",
                "conditions_sufficient": None,
                "supporting_steps": raw.get("supporting_steps", []),
            }
        )
        claims.append(claim.to_dict())
    return {"parse_status": "valid", "claims": claims, "raw_response": content}


def _build_evidence_review_request(
    model: str,
    problem: str,
    checkpoint: str,
    claims: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    *,
    json_mode: bool = False,
) -> dict[str, Any]:
    fields = ("claim_id", "assertion", "conditions", "source_locator", "target_relation")
    assessment_input = [{key: claim[key] for key in fields} for claim in claims]
    instructions = (
        "Independent evidence review:\n"
        "Review each listed claim against the problem and visible checkpoint. "
        "Return only a json object, for example: {\"reviews\": []}. Include one review object per claim, containing "
        "claim_id, verdict (verified, falsified, or unknown), "
        "source_locator (an exact nonempty quote from the checkpoint), "
        "conditions_sufficient (true or false), rationale (checkable mathematical steps "
        "or a counterexample), and milestone (integer 0 to 3). "
        "Use 0 for no confirmed progress, 1 for a relevant local result, "
        "2 for a verified subgoal or materially reduced problem, and 3 for a verified "
        "key chain with only explicit finishing steps left. "
        "Use unknown when the claim or its conditions cannot be checked. "
        "Return JSON even when all claims are unknown.\n"
    )
    request = _build_request(
        model, problem, checkpoint,
        instructions + json.dumps(assessment_input, ensure_ascii=False),
        parameters,
    )
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    return request


def _apply_evidence_review(
    content: str, claims: Sequence[dict[str, Any]], checkpoint: str, response_ref: str
) -> tuple[list[dict[str, Any]], str | None]:
    payload = _parse_json_object(content)
    reviews = payload.get("reviews") if payload is not None else None
    if not isinstance(reviews, list):
        return list(claims), "independent evidence review could not be parsed"
    expected = {claim["claim_id"] for claim in claims}
    by_id = {}
    for row in reviews:
        if not isinstance(row, Mapping) or not isinstance(row.get("claim_id"), str):
            return list(claims), "independent evidence review has invalid claim identities"
        claim_id = row["claim_id"]
        if claim_id not in expected or claim_id in by_id:
            return list(claims), "independent evidence review has invalid claim identities"
        by_id[claim_id] = row
    if by_id.keys() != expected:
        return list(claims), "independent evidence review did not cover every extracted claim"
    assessed = []
    for claim in claims:
        row = by_id[claim["claim_id"]]
        locator, rationale = row.get("source_locator"), row.get("rationale")
        verdict, milestone = row.get("verdict"), row.get("milestone")
        if (
            verdict in ("verified", "falsified")
            and isinstance(locator, str) and locator.strip() and locator in checkpoint
            and isinstance(rationale, str) and rationale.strip()
            and row.get("conditions_sufficient") is True
            and type(milestone) is int and 0 <= milestone <= 3
        ):
            claim = {
                **claim,
                "milestones": [milestone],
                "evidence": [{
                    "provenance": "Agent-Assessed",
                    "verdict": verdict,
                    "source_locator": locator,
                    "conditions": claim["conditions"],
                    "conditions_sufficient": True,
                    "rationale": rationale,
                    "audit_ref": response_ref,
                }],
            }
        assessed.append(claim)
    return assessed, None


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
    evidence_failure: str | None = None,
    json_evidence: bool = False,
) -> dict[str, Any]:
    """Create the persisted pre-rollout score artifact with no future outcomes attached."""
    progress = score_verified_progress(
        verified_claims,
        sample_id=sample_id,
        route_id=route_id,
        execution_failure=evidence_failure,
    )
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "route_id": route_id,
        "problem": problem,
        "checkpoint": checkpoint,
        "templates": {
            "llm_rater": RATER_TEMPLATE_VERSION,
            "evidence": JSON_EVIDENCE_TEMPLATE_VERSION if json_evidence else EVIDENCE_TEMPLATE_VERSION,
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
    review_extracted_claims: bool = False,
    json_evidence: bool = False,
    redispatch_failed: bool = False,
) -> dict[str, Any]:
    """Run both scoring roles through the audited transport using only controlled visible input."""
    parameters = dict(config.model.parameters)
    requests = build_blinded_scoring_requests(
        config.model.model, problem, checkpoint, parameters, json_evidence=json_evidence
    )
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
        redispatch_failed=redispatch_failed,
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
        redispatch_failed=redispatch_failed,
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

    evidence_failure = None
    if review_extracted_claims:
        if extraction["parse_status"] != "valid":
            evidence_failure = "evidence extraction could not be parsed"
        elif extraction["claims"]:
            review_id = _logical_call_id(run_id, sample_id, route_id, "evidence-review")
            review_request = _build_evidence_review_request(
                config.model.model, problem, checkpoint, extraction["claims"], parameters,
                json_mode=json_evidence,
            )
            review_ref = dispatch_request(
                config, ledger, run_dir, review_id, sample_id, review_request,
                effective_parameters=parameters,
                redispatch_failed=redispatch_failed,
            )
            review_payload = read_response(run_dir, review_ref)
            verified_claims, evidence_failure = _apply_evidence_review(
                _response_text(review_payload), extraction["claims"], checkpoint, review_ref
            )
            calls.append({
                "call_id": review_id,
                "sample_id": sample_id,
                "route_id": route_id,
                "stage": "evidence-review",
                "usage": _usage(review_payload),
            })

    artifact = freeze_scoring_artifact(
        sample_id=sample_id,
        route_id=route_id,
        problem=problem,
        checkpoint=checkpoint,
        llm_rater=parsed_rater,
        claim_extraction=extraction,
        verified_claims=verified_claims,
        costs=calls,
        evidence_failure=evidence_failure,
        json_evidence=json_evidence,
    )
    if review_extracted_claims:
        artifact["templates"]["evidence_review"] = (
            JSON_EVIDENCE_REVIEW_TEMPLATE_VERSION if json_evidence else EVIDENCE_REVIEW_TEMPLATE_VERSION
        )
    artifact["artifact_ref"] = str(_artifact_path(run_dir, sample_id, route_id).relative_to(run_dir))
    _write_json(_artifact_path(run_dir, sample_id, route_id), artifact)
    return artifact


# Stable public aliases for orchestration callers.
build_rater_request = build_blinded_rater_request
build_evidence_request = build_blinded_evidence_request
parse_rater_response = parse_llm_rater_response
score_checkpoint = score_ms2_checkpoint
