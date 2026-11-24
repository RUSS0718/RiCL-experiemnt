"""MS2 schemas for frozen inputs and auditable trace exports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MS2Record(BaseModel):
    """One public sample row from a prepared MS2 manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sample_id: str
    problem: str
    source: str | None = None
    revision: str | None = None
    split: str | None = None
    source_record_id: str | None = None
    problem_fingerprint: str | None = None
    question_group_id: str | None = None
    trajectory_id: str | None = None
    trajectory_source_id: str | None = None
    match_basis: str | None = None
    study: str | None = None
    phase: str | None = None
    sampling_stratum: str | None = None
    seed: int | None = None
    evaluation_rule_version: str | None = None

    @field_validator("sample_id", "problem")
    @classmethod
    def require_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("MS2 sample_id and problem must not be empty")
        return value


class MS2Manifest(BaseModel):
    """Versioned MS2 manifest accepted by run and resume commands."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = Field(ge=1)
    manifest_id: str
    study: Literal["ms2"]
    phase: Literal["pilot", "adjustment_validation", "confirmatory"]
    samples: list[MS2Record] = Field(min_length=1)
    rule_version: str | None = None
    identity_rule_version: str | None = None
    sources: list[dict[str, Any]] | None = None
    sampling: dict[str, Any] | None = None
    fingerprint: str | None = None

    @field_validator("samples")
    @classmethod
    def unique_sample_ids(cls, value: list[MS2Record]) -> list[MS2Record]:
        ids = [sample.sample_id for sample in value]
        if len(ids) != len(set(ids)):
            raise ValueError("MS2 sample IDs must be unique")
        return value


class MS2AnswerKeyRow(BaseModel):
    """Restricted answer material never passed to a generation role."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sample_id: str
    reference_answer: str | int | float
    reference_solution: str | None = None
    source_record_id: str | None = None

    @field_validator("sample_id")
    @classmethod
    def require_sample_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer-key sample_id must not be empty")
        return value

    @field_validator("reference_answer")
    @classmethod
    def require_reference(cls, value: str | float) -> str | int | float:
        if isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
            raise ValueError("answer-key reference_answer must not be empty")
        return value


class MS2TraceSegment(BaseModel):
    """A source prefix or independently generated segment in a trace."""

    model_config = ConfigDict(extra="allow", strict=True)

    kind: str
    status: Literal["completed", "failed", "not_dispatched", "unavailable"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    response_ref: str | None = None
    failure: str | None = None


class MS2Rollout(BaseModel):
    """One of the two independent logical continuations for a Checkpoint."""

    model_config = ConfigDict(extra="allow", strict=True)

    rollout_index: int = Field(ge=0, le=1)
    status: Literal["completed", "failed", "not_dispatched"]
    checkpoint_prefix_fingerprint: str
    logical_call_id: str | None = None
    response_ref: str | None = None
    text: str | None = None
    failure: str | None = None
    evaluation_reference: dict[str, Any] | None = None


class MS2CandidateTrace(BaseModel):
    """Candidate identity, immutable visible prefix, initial score, and rollouts."""

    model_config = ConfigDict(extra="allow", strict=True)

    candidate_index: int = Field(ge=0, le=2)
    candidate_id: str
    route_id: str
    problem_id: str
    checkpoint_id: str
    prefix_text: str
    prefix_fingerprint: str
    prefix_status: Literal["available", "empty", "unavailable"]
    truncation_status: Literal["truncated", "not_truncated", "unknown"]
    stop_reason: Any = None
    last_step_status: Literal["complete", "incomplete", "unknown"]
    initial_evaluation: dict[str, Any]
    rollouts: list[MS2Rollout] = Field(min_length=2, max_length=2)
    segments: list[MS2TraceSegment]

    @field_validator("rollouts")
    @classmethod
    def independent_rollout_indices(cls, value: list[MS2Rollout]) -> list[MS2Rollout]:
        if {rollout.rollout_index for rollout in value} != {0, 1}:
            raise ValueError("each MS2 Checkpoint requires rollout indices 0 and 1")
        return value


class MS2Trace(BaseModel):
    """Persisted run export consumed by later scoring and analysis tickets."""

    model_config = ConfigDict(extra="allow", strict=True)

    schema_version: int = Field(ge=1)
    run_id: str
    study: Literal["ms2"]
    phase: Literal["pilot", "adjustment_validation", "confirmatory"]
    manifest_id: str
    manifest_fingerprint: str
    candidate_slots: Literal[3]
    rollouts_per_checkpoint: Literal[2]
    continuation_template_version: str
    samples: list[dict[str, Any]]
    usage: dict[str, int]


def empty_rollout(index: int, prefix_fingerprint: str, reason: str) -> dict[str, Any]:
    """Represent an expected but deliberately un-dispatched rollout."""
    return {
        "rollout_index": index,
        "status": "not_dispatched",
        "checkpoint_prefix_fingerprint": prefix_fingerprint,
        "failure": reason,
        "attempts": [],
        "usage": {
            "status": "not_dispatched",
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
        },
    }


def base_trace(run_id: str, manifest: MS2Manifest, manifest_fingerprint: str) -> dict[str, Any]:
    """Create the top-level JSON export envelope."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "study": "ms2",
        "phase": manifest.phase,
        "manifest_id": manifest.manifest_id,
        "manifest_fingerprint": manifest_fingerprint,
        "candidate_slots": 3,
        "rollouts_per_checkpoint": 2,
        "continuation_template_version": "ms2-neutral-continuation-v1",
        "continuation_instructions": (
            "Inspect the visible work, continue or correct it as needed, "
            "and explicitly state a final answer."
        ),
        "samples": [],
    }
