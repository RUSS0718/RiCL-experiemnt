"""Validated model, budget, and single-sample manifest configuration."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Reject unrecognized fields at persisted input boundaries."""

    model_config = ConfigDict(extra="forbid")


class ModelCapabilities(StrictModel):
    """Static allowlist of parameters supported by one configured endpoint."""

    supported_parameters: list[str]
    required_parameters: list[str] = Field(default_factory=list)

    @field_validator("supported_parameters", "required_parameters")
    @classmethod
    def stable_parameter_order(cls, value: list[str]) -> list[str]:
        """Normalize capability collections so fingerprints are stable across processes."""
        return sorted(set(value))


class ModelSettings(StrictModel):
    """OpenAI-compatible HTTP endpoint and its effective generation settings."""

    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=60, gt=0)
    capabilities: ModelCapabilities
    parameters: dict[str, str | int | float | bool]

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Require a credential-free HTTP base URL so it is safe to summarize."""
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("model.base_url must be an absolute HTTP(S) URL")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("model.base_url must not contain credentials, query, or fragment")
        return value.rstrip("/")

    @field_validator("model", "api_key_env")
    @classmethod
    def validate_nonempty_optional(cls, value: str | None) -> str | None:
        """Reject empty model identifiers and environment variable names."""
        if value is not None and not value.strip():
            raise ValueError("model and api_key_env values must not be empty")
        return value

    @model_validator(mode="after")
    def validate_parameters(self) -> ModelSettings:
        """Fail before dispatch if any configured or required parameter is unsupported."""
        unsupported = self.parameters.keys() - set(self.capabilities.supported_parameters)
        missing = set(self.capabilities.required_parameters) - self.parameters.keys()
        if unsupported:
            raise ValueError(f"unsupported configured model parameters: {', '.join(sorted(unsupported))}")
        if missing:
            raise ValueError(f"missing required model parameters: {', '.join(sorted(missing))}")
        if "max_tokens" in self.parameters:
            max_tokens = self.parameters["max_tokens"]
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
                raise ValueError("model.parameters.max_tokens must be a positive integer")
        return self


class BudgetSettings(StrictModel):
    """Per-logical-call send limit and cumulative output-token budget."""

    max_attempts_per_call: int = Field(ge=1)
    max_reserved_output_tokens: int = Field(ge=1)


class RunConfig(StrictModel):
    """Complete frozen inputs for a model-backed run."""

    model: ModelSettings
    budget: BudgetSettings

    @model_validator(mode="after")
    def validate_budget_covers_request(self) -> RunConfig:
        """Require the token budget to cover at least one configured attempt."""
        max_tokens = self.model.parameters.get("max_tokens")
        if max_tokens is None:
            raise ValueError("model.parameters.max_tokens is required for bounded token reservations")
        if self.budget.max_reserved_output_tokens < max_tokens:
            raise ValueError("budget must cover one max_tokens reservation")
        return self

    def redacted(self) -> dict[str, Any]:
        """Return serializable configuration metadata without credential values."""
        return self.model_dump(mode="json")


class Sample(StrictModel):
    """One generation-eligible sample; labels and answer keys are not accepted."""

    sample_id: str
    prompt: str

    @field_validator("sample_id", "prompt")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        """Reject empty identities and generation prompts."""
        if not value.strip():
            raise ValueError("sample_id and prompt must not be empty")
        return value


class SampleManifest(StrictModel):
    """Versioned, intentionally single-sample input for the first runner slice."""

    schema_version: int = Field(ge=1)
    manifest_id: str
    dataset_version: str
    samples: list[Sample] = Field(min_length=1, max_length=1)


class InputDriftError(ValueError):
    """Raised when a resume no longer matches its frozen run inputs."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for fingerprints and audit records."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    """Hash a JSON-compatible value with SHA-256."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_config(path: Path) -> RunConfig:
    """Read and validate the configured TOML model and budget."""
    with path.open("rb") as file:
        return RunConfig.model_validate(tomllib.load(file))


def load_manifest(path: Path) -> SampleManifest:
    """Read and validate the versioned JSON sample manifest."""
    return SampleManifest.model_validate_json(path.read_text(encoding="utf-8"))


def validate_resume_inputs(
    config: RunConfig,
    manifest: SampleManifest,
    metadata: dict[str, Any],
) -> None:
    """Reject configuration, manifest, or prompt changes for an existing run."""
    actual = {
        "config_fingerprint": fingerprint(config.redacted()),
        "manifest_fingerprint": fingerprint(manifest.model_dump(mode="json")),
        "prompt_fingerprint": fingerprint([sample.prompt for sample in manifest.samples]),
    }
    for key, value in actual.items():
        if metadata.get(key) != value:
            raise InputDriftError(f"resume rejected: {key} does not match the original run")


def validate_api_key(config: RunConfig) -> str | None:
    """Resolve a configured credential from the environment without persisting it."""
    name = config.model.api_key_env
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value
