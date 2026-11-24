"""Frozen MS3 repair groups, prompt isolation, and blind evaluation records."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

from ..core.model import build_request_body

GROUPS = ("G0", "G1", "G2", "G3")
LOCATION_TEMPLATE = "请优先复核第 {step_index} 步及其依据。"
_GENERIC_REPAIR = "检查原轨迹的推理并修复。"
_FINAL_REQUIREMENT = "请给出可独立评估的完整修复解答，并明确写出结论。"
GROUP_INSTRUCTIONS = {
    "G0": f"根据原题重新求解。输入仍包含同一原始轨迹。{_FINAL_REQUIREMENT}",
    "G1": f"{_GENERIC_REPAIR}{_FINAL_REQUIREMENT}",
    "G2": f"{_GENERIC_REPAIR}{_FINAL_REQUIREMENT}",
    "G3": f"{_GENERIC_REPAIR}{_FINAL_REQUIREMENT}",
}


def build_repair_request(
    problem: str,
    steps: list[str] | tuple[str, ...],
    group: str,
    location_index: int | None,
    parameters: Mapping[str, Any] | None = None,
    model: str = "model",
) -> dict[str, Any]:
    """Build one generation request from the intervention's explicit allowlist."""
    if group not in GROUPS:
        raise ValueError(f"unknown MS3 repair group: {group}")
    if group in {"G2", "G3"} and location_index is None:
        raise ValueError(f"{group} requires a location index")
    if group in {"G0", "G1"} and location_index is not None:
        raise ValueError(f"{group} cannot receive a location index")
    instruction = GROUP_INSTRUCTIONS[group]
    if location_index is not None:
        instruction = f"{instruction}\n{LOCATION_TEMPLATE.format(step_index=location_index)}"
    indexed_steps = [f"{index}: {step}" for index, step in enumerate(steps)]
    request = build_request_body(
        model,
        "ms3_repair",
        {"problem": problem, "steps": indexed_steps, "repair_instruction": instruction},
        dict(parameters or {}),
    )
    request["tools"] = []
    return request


def choose_random_location(seed: int, trajectory_id: str, step_count: int, first_error: int) -> int:
    """Draw one deterministic uniform non-first-error zero-based step index."""
    if step_count < 2 or first_error < 0 or first_error >= step_count:
        raise ValueError("random location requires at least two steps and a valid first error")
    candidates = [index for index in range(step_count) if index != first_error]
    randomizer = random.Random(_location_seed(seed, trajectory_id))
    return randomizer.choice(candidates)


def validate_repair_trajectory(
    steps: list[str] | tuple[str, ...], first_error: int
) -> tuple[bool, str | None]:
    """Apply the shared MS3 repair eligibility rule before four-group dispatch."""
    if len(steps) < 2 or any(not isinstance(step, str) or not step.strip() for step in steps):
        return False, "insufficient_steps"
    if isinstance(first_error, bool) or not isinstance(first_error, int):
        return False, "invalid_first_error_position"
    if first_error < 0 or first_error >= len(steps):
        return False, "invalid_first_error_position"
    return True, None


def evaluate_repair_record(
    evaluation_id: str,
    repair_success: str,
    error_removal: str,
    new_error: str,
    evidence: Mapping[str, str] | None = None,
    missing_reason: str | None = None,
) -> dict[str, Any]:
    """Validate one blinded human decision with independent outcome dimensions."""
    allowed = {"correct", "incorrect", "unknown", "missing"}
    if repair_success not in allowed:
        raise ValueError("repair_success must be correct, incorrect, unknown, or missing")
    if error_removal not in {"removed", "not_removed", "unknown", "missing"}:
        raise ValueError("invalid error_removal outcome")
    if new_error not in {"present", "absent", "unknown", "missing"}:
        raise ValueError("invalid new_error outcome")
    if evidence is not None and any(not isinstance(value, str) for value in evidence.values()):
        raise ValueError("evaluation evidence values must be text")
    return {
        "evaluation_id": evaluation_id,
        "repair_success": repair_success,
        "error_removal": error_removal,
        "new_error": new_error,
        "evidence": dict(evidence or {}),
        "missing_reason": missing_reason,
    }


def _location_seed(seed: int, trajectory_id: str) -> int:
    """Derive a stable pseudo-random seed from the frozen sampling inputs."""
    digest = hashlib.sha256(f"{seed}:{trajectory_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
