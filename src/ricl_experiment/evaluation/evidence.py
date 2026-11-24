"""Auditable MS2 claims, evidence provenance, scoring, cost, and blind review."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class EvidenceProvenance(StrEnum):
    """Frozen evidence-source vocabulary from the MS2 scoring decision."""

    TOOL_VERIFIED = "Tool-Verified"
    DERIVATION_CHECKED = "Derivation-Checked"
    AGENT_ASSESSED = "Agent-Assessed"
    UNKNOWN = "Unknown"


class ClaimVerdict(StrEnum):
    """Mathematical status assigned to one deduplicated claim."""

    VERIFIED = "verified"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"


EvidenceSource = EvidenceProvenance
MathematicalVerdict = ClaimVerdict

_SOURCE_PRIORITY = {
    EvidenceProvenance.TOOL_VERIFIED: 0,
    EvidenceProvenance.DERIVATION_CHECKED: 1,
    EvidenceProvenance.AGENT_ASSESSED: 2,
    EvidenceProvenance.UNKNOWN: 3,
}


def _canonical_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _text_tuple(values: Iterable[Any] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values if str(value).strip())


def _source(value: EvidenceProvenance | str | None) -> EvidenceProvenance:
    if isinstance(value, EvidenceProvenance):
        return value
    if value is None:
        return EvidenceProvenance.UNKNOWN
    normalized = str(value).strip().casefold().replace("_", "-")
    for item in EvidenceProvenance:
        if normalized in {item.value.casefold(), item.name.casefold().replace("_", "-")}:
            return item
    raise ValueError(f"unknown evidence provenance: {value}")


def _verdict(value: ClaimVerdict | str | None) -> ClaimVerdict:
    if isinstance(value, ClaimVerdict):
        return value
    if value is None:
        return ClaimVerdict.UNKNOWN
    normalized = str(value).strip().casefold().replace("_", "-")
    aliases = {
        "confirmed": ClaimVerdict.VERIFIED,
        "valid": ClaimVerdict.VERIFIED,
        "refuted": ClaimVerdict.FALSIFIED,
        "invalid": ClaimVerdict.FALSIFIED,
        "unresolved": ClaimVerdict.UNKNOWN,
    }
    if normalized in aliases:
        return aliases[normalized]
    for item in ClaimVerdict:
        if normalized in {item.value, item.name.casefold()}:
            return item
    raise ValueError(f"unknown claim verdict: {value}")


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class EvidenceRecord:
    """One independently traceable piece of mathematical evidence."""

    provenance: EvidenceProvenance
    verdict: ClaimVerdict
    source_locator: str = ""
    conditions: tuple[str, ...] = ()
    conditions_sufficient: bool | None = True
    supporting_steps: tuple[str, ...] = ()
    rationale: str = ""
    tool_name: str | None = None
    tool_version: str | None = None
    audit_ref: str | None = None
    checked_at: str | None = None
    raw_result: Any = None

    def __post_init__(self) -> None:
        if not self.source_locator.strip() and self.provenance != EvidenceProvenance.UNKNOWN:
            raise ValueError("non-Unknown evidence requires a source locator")
        if self.conditions_sufficient not in {True, False, None}:
            raise ValueError("conditions_sufficient must be true, false, or null")

    @property
    def usable(self) -> bool:
        return self.conditions_sufficient is True

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.value,
            "verdict": self.verdict.value,
            "source_locator": self.source_locator,
            "conditions": list(self.conditions),
            "conditions_sufficient": self.conditions_sufficient,
            "supporting_steps": list(self.supporting_steps),
            "rationale": self.rationale,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "audit_ref": self.audit_ref,
            "checked_at": self.checked_at,
            "raw_result": _json_value(self.raw_result),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvidenceRecord:
        return cls(
            provenance=_source(value.get("provenance", value.get("source"))),
            verdict=_verdict(value.get("verdict", value.get("status"))),
            source_locator=str(value.get("source_locator", value.get("location", ""))),
            conditions=_text_tuple(value.get("conditions", value.get("applicable_conditions"))),
            conditions_sufficient=value.get("conditions_sufficient", True),
            supporting_steps=_text_tuple(value.get("supporting_steps", value.get("support"))),
            rationale=str(value.get("rationale", value.get("reason", ""))),
            tool_name=value.get("tool_name"),
            tool_version=value.get("tool_version"),
            audit_ref=value.get("audit_ref"),
            checked_at=value.get("checked_at"),
            raw_result=value.get("raw_result"),
        )


@dataclass(frozen=True)
class ClaimRecord:
    """A deduplicatable mathematical claim with all supporting evidence retained."""

    claim_id: str
    sample_id: str
    route_id: str
    problem: str
    checkpoint: str
    assertion: str
    conditions: tuple[str, ...] = ()
    target_relation: str = ""
    source_locator: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()
    equivalence_key: str | None = None
    nontrivial: bool = True
    target_relevant: bool = True
    milestones: tuple[int, ...] = (1,)
    claim_variants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.assertion.strip():
            raise ValueError("claim_id and assertion must not be empty")
        if any(level < 0 or level > 3 for level in self.milestones):
            raise ValueError("milestone levels must be between 0 and 3")

    @property
    def eligible(self) -> bool:
        return self.nontrivial and self.target_relevant

    @property
    def canonical_equivalence_key(self) -> str:
        return self.equivalence_key or _canonical_text(self.assertion)

    @property
    def resolved_verdict(self) -> ClaimVerdict:
        usable = [item.verdict for item in self.evidence if item.usable]
        concrete = {item for item in usable if item != ClaimVerdict.UNKNOWN}
        if len(concrete) != 1:
            return ClaimVerdict.UNKNOWN
        return next(iter(concrete))

    @property
    def primary_provenance(self) -> EvidenceProvenance:
        evidence = [item for item in self.evidence if item.usable] or list(self.evidence)
        if not evidence:
            return EvidenceProvenance.UNKNOWN
        return min((item.provenance for item in evidence), key=_SOURCE_PRIORITY.__getitem__)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "sample_id": self.sample_id,
            "route_id": self.route_id,
            "problem": self.problem,
            "checkpoint": self.checkpoint,
            "assertion": self.assertion,
            "conditions": list(self.conditions),
            "target_relation": self.target_relation,
            "source_locator": self.source_locator,
            "equivalence_key": self.canonical_equivalence_key,
            "nontrivial": self.nontrivial,
            "target_relevant": self.target_relevant,
            "milestones": list(self.milestones),
            "claim_variants": list(self.claim_variants),
            "evidence": [item.to_dict() for item in self.evidence],
            "verdict": self.resolved_verdict.value,
            "primary_provenance": self.primary_provenance.value,
            "eligible": self.eligible,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClaimRecord:
        assertion = str(value.get("assertion", value.get("claim", value.get("claim_text", ""))))
        sample_id = str(value.get("sample_id", ""))
        route_id = str(value.get("route_id", ""))
        conditions = _text_tuple(value.get("conditions", value.get("applicable_conditions")))
        evidence_value = value.get("evidence")
        if isinstance(evidence_value, Sequence) and not isinstance(evidence_value, (str, bytes)):
            evidence = tuple(
                item if isinstance(item, EvidenceRecord) else EvidenceRecord.from_mapping(item)
                for item in evidence_value
            )
        else:
            evidence = (
                (
                    EvidenceRecord.from_mapping(value),
                )
                if any(key in value for key in ("provenance", "source", "verdict", "status"))
                else ()
            )
        milestones = value.get("milestones", value.get("milestone", (1,)))
        if isinstance(milestones, int) and not isinstance(milestones, bool):
            milestones = (milestones,)
        claim_id = str(value.get("claim_id", ""))
        if not claim_id:
            claim_id = hashlib.sha256(
                json.dumps(
                    [sample_id, route_id, assertion, list(conditions), value.get("source_locator", "")],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24]
        return cls(
            claim_id=claim_id,
            sample_id=sample_id,
            route_id=route_id,
            problem=str(value.get("problem", "")),
            checkpoint=str(value.get("checkpoint", "")),
            assertion=assertion,
            conditions=conditions,
            target_relation=str(value.get("target_relation", value.get("relation", ""))),
            source_locator=str(value.get("source_locator", value.get("location", ""))),
            evidence=evidence,
            equivalence_key=value.get("equivalence_key"),
            nontrivial=bool(value.get("nontrivial", True)),
            target_relevant=bool(value.get("target_relevant", True)),
            milestones=tuple(int(item) for item in milestones),
            claim_variants=_text_tuple(value.get("claim_variants")),
        )


def coerce_claim(value: ClaimRecord | Mapping[str, Any]) -> ClaimRecord:
    return value if isinstance(value, ClaimRecord) else ClaimRecord.from_mapping(value)


def deduplicate_claims(claims: Iterable[ClaimRecord | Mapping[str, Any]]) -> list[ClaimRecord]:
    """Merge equivalent claims only within one route and one applicability condition set."""
    merged: dict[tuple[Any, ...], ClaimRecord] = {}
    order: list[tuple[Any, ...]] = []
    for raw_claim in claims:
        claim = coerce_claim(raw_claim)
        key = (
            claim.sample_id,
            claim.route_id,
            claim.canonical_equivalence_key,
            tuple(_canonical_text(item) for item in claim.conditions),
        )
        if key not in merged:
            merged[key] = claim
            order.append(key)
            continue
        previous = merged[key]
        evidence = list(previous.evidence)
        seen_evidence = {json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) for item in evidence}
        for item in claim.evidence:
            encoded = json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False)
            if encoded not in seen_evidence:
                evidence.append(item)
                seen_evidence.add(encoded)
        variants = list(previous.claim_variants)
        if claim.assertion != previous.assertion and claim.assertion not in variants:
            variants.append(claim.assertion)
        variants.extend(item for item in claim.claim_variants if item not in variants)
        merged[key] = ClaimRecord(
            claim_id=previous.claim_id,
            sample_id=previous.sample_id,
            route_id=previous.route_id,
            problem=previous.problem or claim.problem,
            checkpoint=previous.checkpoint or claim.checkpoint,
            assertion=previous.assertion,
            conditions=previous.conditions,
            target_relation=previous.target_relation or claim.target_relation,
            source_locator=previous.source_locator or claim.source_locator,
            evidence=tuple(evidence),
            equivalence_key=previous.canonical_equivalence_key,
            nontrivial=previous.nontrivial and claim.nontrivial,
            target_relevant=previous.target_relevant and claim.target_relevant,
            milestones=tuple(sorted(set(previous.milestones) | set(claim.milestones))),
            claim_variants=tuple(variants),
        )
    return [merged[key] for key in order]


build_claim_ledger = deduplicate_claims



def _source_summary(claims: Sequence[ClaimRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[EvidenceProvenance, list[ClaimRecord]] = defaultdict(list)
    for claim in claims:
        grouped[claim.primary_provenance].append(claim)
    report: dict[str, dict[str, Any]] = {}
    for source in EvidenceProvenance:
        rows = grouped.get(source, [])
        counts = {verdict.value: 0 for verdict in ClaimVerdict}
        for claim in rows:
            counts[claim.resolved_verdict.value] += 1
        denominator = len(rows)
        verified_or_falsified = counts[ClaimVerdict.VERIFIED.value] + counts[ClaimVerdict.FALSIFIED.value]
        report[source.value] = {
            "claims": denominator,
            "verified": counts[ClaimVerdict.VERIFIED.value],
            "falsified": counts[ClaimVerdict.FALSIFIED.value],
            "unknown": counts[ClaimVerdict.UNKNOWN.value],
            "coverage": verified_or_falsified / denominator if denominator else None,
        }
    return report


def score_verified_progress(
    claims: Iterable[ClaimRecord | Mapping[str, Any]],
    *,
    sample_id: str | None = None,
    route_id: str | None = None,
    execution_failure: str | None = None,
) -> dict[str, Any]:
    """Score the highest audited milestone and report the claim coverage denominator."""
    normalized = deduplicate_claims(claims)
    if execution_failure is not None:
        return {
            "sample_id": sample_id,
            "route_id": route_id,
            "score": None,
            "coverage": None,
            "coverage_status": "missing",
            "failure_reason": execution_failure,
            "coverage_denominator": 0,
            "coverage_numerator": 0,
            "eligible_claims": 0,
            "verified": 0,
            "falsified": 0,
            "unknown": 0,
            "evidence_by_source": _source_summary([]),
        }
    eligible = [claim for claim in normalized if claim.eligible]
    counts = {
        verdict: sum(claim.resolved_verdict == verdict for claim in eligible)
        for verdict in ClaimVerdict
    }
    denominator = len(eligible)
    numerator = counts[ClaimVerdict.VERIFIED] + counts[ClaimVerdict.FALSIFIED]
    supported = [claim for claim in eligible if claim.resolved_verdict == ClaimVerdict.VERIFIED]
    supported_levels = {
        level for claim in supported for level in claim.milestones if 0 <= level <= 3
    }
    score = max(supported_levels, default=0)
    coverage_status = (
        "not_applicable" if not eligible else "unverified" if numerator == 0 else "observed"
    )
    score_basis = []
    for claim in supported:
        for evidence in claim.evidence:
            if evidence.usable and evidence.verdict == ClaimVerdict.VERIFIED:
                score_basis.extend(
                    {
                        "claim_id": claim.claim_id,
                        "milestone": level,
                        "source_locator": evidence.source_locator,
                        "conditions": list(claim.conditions or evidence.conditions),
                        "rationale": evidence.rationale,
                        "provenance": evidence.provenance.value,
                    }
                    for level in claim.milestones
                    if level in supported_levels
                )
    return {
        "sample_id": sample_id,
        "route_id": route_id,
        "score": score,
        "coverage": numerator / denominator if denominator else None,
        "coverage_status": coverage_status,
        "coverage_denominator": denominator,
        "coverage_numerator": numerator,
        "eligible_claims": denominator,
        "verified": counts[ClaimVerdict.VERIFIED],
        "falsified": counts[ClaimVerdict.FALSIFIED],
        "unknown": counts[ClaimVerdict.UNKNOWN],
        "unverified": bool(eligible) and numerator == 0,
        "milestones_supported": sorted(supported_levels),
        "score_basis": score_basis,
        "evidence_by_source": _source_summary(eligible),
        "claims": [claim.to_dict() for claim in normalized],
    }


score_milestones = score_verified_progress


@dataclass(frozen=True)
class CallCost:
    """Auditable usage for one scoring or review call."""

    call_id: str
    sample_id: str | None = None
    route_id: str | None = None
    stage: str | None = None
    usage_status: str = "unknown"
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CallCost:
        usage = value.get("usage") if isinstance(value.get("usage"), Mapping) else value
        return cls(
            call_id=str(value.get("call_id", value.get("logical_call_id", ""))),
            sample_id=value.get("sample_id"),
            route_id=value.get("route_id"),
            stage=value.get("stage", value.get("role")),
            usage_status=str(usage.get("status", value.get("usage_status", "unknown"))),
            input_tokens=usage.get("input_tokens", usage.get("prompt_tokens")),
            output_tokens=usage.get("output_tokens", usage.get("completion_tokens")),
            reasoning_tokens=usage.get("reasoning_tokens"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "sample_id": self.sample_id,
            "route_id": self.route_id,
            "stage": self.stage,
            "usage_status": self.usage_status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


def _cost_record(value: CallCost | Mapping[str, Any]) -> CallCost:
    return value if isinstance(value, CallCost) else CallCost.from_mapping(value)


def _empty_cost() -> dict[str, Any]:
    return {
        "calls": 0,
        "known_usage_calls": 0,
        "unknown_usage_calls": 0,
        "tokens": {"input": 0, "output": 0, "reasoning": 0},
        "unknown_token_calls": {"input": 0, "output": 0, "reasoning": 0},
        "cost": 0.0,
        "cost_status": "empty",
    }


def _add_cost(
    target: dict[str, Any],
    record: CallCost,
    *,
    input_rate: float,
    output_rate: float,
    reasoning_rate: float,
) -> None:
    target["calls"] += 1
    known = record.usage_status == "known"
    target["known_usage_calls"] += int(known)
    target["unknown_usage_calls"] += int(not known)
    rates = {"input": input_rate, "output": output_rate, "reasoning": reasoning_rate}
    values = {
        "input": record.input_tokens,
        "output": record.output_tokens,
        "reasoning": record.reasoning_tokens,
    }
    for name, value in values.items():
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            target["tokens"][name] += value
            target["cost"] += value * rates[name] / 1000
        else:
            target["unknown_token_calls"][name] += 1
    if target["unknown_usage_calls"]:
        target["cost_status"] = "partial" if target["known_usage_calls"] else "unknown"
    elif any(target["unknown_token_calls"][name] and rates[name] for name in rates):
        target["cost_status"] = "partial"
    else:
        target["cost_status"] = "known"


def summarize_costs(
    records: Iterable[CallCost | Mapping[str, Any]],
    *,
    input_cost_per_1k: float = 0.0,
    output_cost_per_1k: float = 0.0,
    reasoning_cost_per_1k: float = 0.0,
) -> dict[str, Any]:
    """Aggregate usage by route, sample, and overall without filling unknown usage."""
    rates = (input_cost_per_1k, output_cost_per_1k, reasoning_cost_per_1k)
    if any(rate < 0 for rate in rates):
        raise ValueError("cost rates must not be negative")
    overall = _empty_cost()
    by_route: dict[str, dict[str, Any]] = {}
    by_sample: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        record = _cost_record(raw_record)
        kwargs = {
            "input_rate": input_cost_per_1k,
            "output_rate": output_cost_per_1k,
            "reasoning_rate": reasoning_cost_per_1k,
        }
        _add_cost(overall, record, **kwargs)
        route = str(record.route_id or "unknown")
        sample = str(record.sample_id or "unknown")
        by_route.setdefault(route, _empty_cost())
        by_sample.setdefault(sample, _empty_cost())
        _add_cost(by_route[route], record, **kwargs)
        _add_cost(by_sample[sample], record, **kwargs)
    for group in (overall, *by_route.values(), *by_sample.values()):
        group["cost"] = round(group["cost"], 10)
    return {"overall": overall, "by_route": by_route, "by_sample": by_sample}


summarize_call_costs = summarize_costs


def _read_json(value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | Path | str) -> Any:
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (Path, str)) else value


def _write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _stratum_seed(seed: int, key: tuple[str, str]) -> int:
    digest = hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_blind_review_package(
    claims: Iterable[ClaimRecord | Mapping[str, Any]], *, seed: int, sample_size: int = 24
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a reviewer-visible task file and a separate restricted audit mapping."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    population = [claim for claim in deduplicate_claims(claims) if claim.eligible]
    strata: dict[tuple[str, str], list[ClaimRecord]] = defaultdict(list)
    for claim in population:
        strata[(claim.primary_provenance.value, claim.resolved_verdict.value)].append(claim)
    shuffled: dict[tuple[str, str], list[ClaimRecord]] = {}
    for key, rows in strata.items():
        ordered = sorted(rows, key=lambda claim: claim.claim_id)
        random.Random(_stratum_seed(seed, key)).shuffle(ordered)
        shuffled[key] = ordered
    selected: list[tuple[tuple[str, str], ClaimRecord]] = []
    target = min(sample_size, len(population))
    while len(selected) < target:
        progressed = False
        for key in sorted(shuffled):
            if shuffled[key]:
                selected.append((key, shuffled[key].pop(0)))
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
    tasks: list[dict[str, Any]] = []
    claim_map: dict[str, str] = {}
    selected_strata: dict[str, dict[str, str]] = {}
    for key, claim in selected:
        review_id = hashlib.sha256(
            f"ms2-blind-review-v1:{seed}:{claim.claim_id}".encode()
        ).hexdigest()[:24]
        tasks.append(
            {
                "review_id": review_id,
                "problem": claim.problem,
                "checkpoint": claim.checkpoint,
                "claim": claim.assertion,
                "conditions": list(claim.conditions),
                "target_relation": claim.target_relation,
                "source_locator": claim.source_locator,
                "supporting_steps": [step for item in claim.evidence for step in item.supporting_steps],
            }
        )
        claim_map[review_id] = claim.claim_id
        selected_strata[review_id] = {"provenance": key[0], "verdict": key[1]}
    public = {
        "schema_version": 1,
        "kind": "ms2-blind-math-review",
        "seed": seed,
        "requested_count": sample_size,
        "selected_count": len(tasks),
        "shortfall": max(0, sample_size - len(tasks)),
        "tasks": tasks,
    }
    audit = {
        "schema_version": 1,
        "kind": "ms2-blind-math-review-audit",
        "seed": seed,
        "requested_count": sample_size,
        "claim_map": claim_map,
        "selected_strata": selected_strata,
        "population_count": len(population),
        "strata_counts": {f"{key[0]}|{key[1]}": len(rows) for key, rows in sorted(strata.items())},
    }
    return public, audit


def export_blind_review(
    claims: Iterable[ClaimRecord | Mapping[str, Any]],
    destination: Path | str | None = None,
    *,
    seed: int,
    sample_size: int = 24,
    audit_destination: Path | str | None = None,
) -> dict[str, Any]:
    """Export only the reviewer-visible task file; persist the audit map separately."""
    public, audit = build_blind_review_package(claims, seed=seed, sample_size=sample_size)
    if destination is not None:
        _write_json(destination, public)
    if audit_destination is not None:
        _write_json(audit_destination, audit)
    return public


sample_blind_claims = build_blind_review_package


def import_blind_reviews(
    reviews: Mapping[str, Any] | Sequence[Mapping[str, Any]] | Path | str,
    *,
    public_export: Mapping[str, Any] | Path | str | None = None,
) -> list[dict[str, Any]]:
    """Validate controlled reviewer records against the exported blind task IDs."""
    payload = _read_json(reviews)
    rows = payload.get("reviews", []) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("reviews must be a list or an object containing reviews")
    allowed_ids = None
    if public_export is not None:
        public = _read_json(public_export)
        allowed_ids = {str(task["review_id"]) for task in public.get("tasks", [])}
    imported: list[dict[str, Any]] = []
    seen: set[str] = set()
    forbidden_review_keys = {
        "answer",
        "reference_answer",
        "reference_solution",
        "expert_label",
        "score",
        "provenance",
        "claim_id",
        "route_id",
        "sample_id",
        "rollout",
        "rollouts",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each review must be an object")
        leaked = {str(key).casefold() for key in row} & forbidden_review_keys
        if leaked:
            raise ValueError("blind review contains restricted fields")
        review_id = str(row.get("review_id", ""))
        if not review_id or review_id in seen:
            raise ValueError("review IDs must be non-empty and unique")
        if allowed_ids is not None and review_id not in allowed_ids:
            raise ValueError(f"review ID is not present in the blind export: {review_id}")
        imported.append(
            {
                "review_id": review_id,
                "verdict": _verdict(row.get("verdict", row.get("decision"))).value,
                "rationale": str(row.get("rationale", row.get("reason", ""))),
                "reviewer_id": row.get("reviewer_id"),
            }
        )
        seen.add(review_id)
    return imported


def _review_bucket() -> dict[str, Any]:
    return {
        "sampled": 0,
        "adjudicable": 0,
        "agreement": 0,
        "disagreement": 0,
        "manual_unresolved": 0,
        "original_unknown": 0,
        "missing_review": 0,
        "accuracy": None,
    }


def _finish_review_bucket(bucket: dict[str, Any]) -> None:
    if bucket["adjudicable"]:
        bucket["accuracy"] = bucket["agreement"] / bucket["adjudicable"]


def recompute_blind_review(
    claims: Iterable[ClaimRecord | Mapping[str, Any]],
    audit: Mapping[str, Any] | Path | str,
    reviews: Mapping[str, Any] | Sequence[Mapping[str, Any]] | Path | str,
) -> dict[str, Any]:
    """Recompute reviewer agreement and adjudicable denominator by evidence source."""
    audit_payload = _read_json(audit)
    if not isinstance(audit_payload, Mapping) or not isinstance(audit_payload.get("claim_map"), Mapping):
        raise TypeError("audit must contain a claim_map")
    claim_index = {claim.claim_id: claim for claim in deduplicate_claims(claims)}
    review_index = {
        row["review_id"]: row
        for row in import_blind_reviews(
            reviews,
            public_export={"tasks": [{"review_id": review_id} for review_id in audit_payload["claim_map"]]},
        )
    }
    by_source: dict[str, dict[str, Any]] = {}
    overall = _review_bucket()
    for review_id, claim_id in sorted(audit_payload["claim_map"].items()):
        claim = claim_index.get(str(claim_id))
        if claim is None:
            raise ValueError(f"audit references an unknown claim: {claim_id}")
        source = claim.primary_provenance.value
        source_bucket = by_source.setdefault(source, _review_bucket())
        review = review_index.get(str(review_id))
        for bucket in (overall, source_bucket):
            bucket["sampled"] += 1
        if review is None:
            overall["missing_review"] += 1
            source_bucket["missing_review"] += 1
            continue
        original = claim.resolved_verdict
        reviewer = _verdict(review["verdict"])
        if original == ClaimVerdict.UNKNOWN:
            overall["original_unknown"] += 1
            source_bucket["original_unknown"] += 1
        if reviewer == ClaimVerdict.UNKNOWN:
            overall["manual_unresolved"] += 1
            source_bucket["manual_unresolved"] += 1
        if original == ClaimVerdict.UNKNOWN or reviewer == ClaimVerdict.UNKNOWN:
            continue
        overall["adjudicable"] += 1
        source_bucket["adjudicable"] += 1
        if original == reviewer:
            overall["agreement"] += 1
            source_bucket["agreement"] += 1
        else:
            overall["disagreement"] += 1
            source_bucket["disagreement"] += 1
    _finish_review_bucket(overall)
    for bucket in by_source.values():
        _finish_review_bucket(bucket)
    return {
        "schema_version": 1,
        "seed": audit_payload.get("seed"),
        "requested_count": audit_payload.get("requested_count", audit_payload.get("selected_count")),
        "selected_count": len(audit_payload["claim_map"]),
        "overall": overall,
        "by_provenance": {key: by_source[key] for key in sorted(by_source)},
    }


# Stable public aliases for callers that use the protocol terms.
export_math_review = export_blind_review
import_math_review = import_blind_reviews
recompute_math_review = recompute_blind_review
