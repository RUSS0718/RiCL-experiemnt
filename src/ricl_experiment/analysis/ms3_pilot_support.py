"""Shared artifact, ledger, and planning helpers for the MS3 acceptance gate."""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Z_95 = 1.96
DEFAULT_CONSERVATIVE_SD_MULTIPLIER = 1.5
DEFAULT_CONSERVATIVE_RATE_FACTOR = 0.8


def _build_planning(
    repair_analysis: Mapping[str, Any],
    diagnosis_run: Path,
    repair_run: Path,
    pricing: Mapping[str, float] | None,
    target_half_width: float,
) -> dict[str, Any]:
    """Create transparent pilot-rate, paired-variance, and cost planning outputs."""
    if not isinstance(target_half_width, (int, float)) or target_half_width <= 0:
        raise ValueError("target_half_width must be positive")
    comparisons = {}
    for name in ("G3-G1", "G3-G2", "G1-G0"):
        comparison = repair_analysis.get("comparisons", {}).get(name, {})
        pairs = comparison.get("pairs", [])
        differences = [float(row["difference"]) for row in pairs if "difference" in row]
        observed_sd = statistics.stdev(differences) if len(differences) >= 2 else None
        fallback = observed_sd is None or observed_sd == 0
        planning_sd = observed_sd if observed_sd and observed_sd > 0 else 0.5
        rate = _number_or_none(comparison.get("evaluable_rate"))
        base_evaluable = _required_pairs(planning_sd, target_half_width)
        conservative_evaluable = _required_pairs(
            planning_sd * DEFAULT_CONSERVATIVE_SD_MULTIPLIER, target_half_width
        )
        conservative_rate = rate * DEFAULT_CONSERVATIVE_RATE_FACTOR if rate else None
        comparisons[name] = {
            "observed_candidate_pairs": comparison.get("candidate_pairs", 0),
            "observed_evaluable_pairs": comparison.get("n_pairs", 0),
            "observed_evaluable_rate": rate,
            "observed_difference_sd": observed_sd,
            "planning_sd": planning_sd,
            "planning_sd_source": "conservative_binary_fallback_0.5" if fallback else "pilot_sample_sd",
            "target_half_width": target_half_width,
            "base_required_evaluable_pairs": base_evaluable,
            "base_required_candidate_pairs": _inflate_for_rate(base_evaluable, rate),
            "conservative_required_evaluable_pairs": conservative_evaluable,
            "conservative_required_candidate_pairs": _inflate_for_rate(
                conservative_evaluable, conservative_rate
            ),
        }
    return {
        "schema_version": 1,
        "status": "estimated_from_recorded_tokens" if pricing is not None else "pricing_not_provided",
        "sample_size": {
            "confidence_level": 0.95,
            "z_value": Z_95,
            "comparisons": comparisons,
            "assumptions": {
                "conservative_sd_multiplier": DEFAULT_CONSERVATIVE_SD_MULTIPLIER,
                "conservative_rate_factor": DEFAULT_CONSERVATIVE_RATE_FACTOR,
                "fallback_sd": 0.5,
                "unit": "independent error trajectory / paired repair candidate",
            },
        },
        "cost": _cost_plan(diagnosis_run, repair_run, pricing),
    }


def _cost_plan(
    diagnosis_run: Path, repair_run: Path, pricing: Mapping[str, float] | None
) -> dict[str, Any]:
    """Summarize observed ledger tokens and calculate cost only with supplied prices."""
    diagnosis = _usage_summary(Path(diagnosis_run) / "run.sqlite3")
    repair = _usage_summary(Path(repair_run) / "run.sqlite3")
    total = {
        key: diagnosis[key] + repair[key]
        for key in ("attempts", "input_tokens", "output_tokens", "reasoning_tokens", "unknown_usage_attempts")
    }
    result: dict[str, Any] = {"observed": {"diagnosis": diagnosis, "repair": repair, "total": total}}
    if pricing is None:
        result["status"] = "pricing_not_provided"
        return result
    input_rate = float(pricing.get("input_per_1k_tokens", -1))
    output_rate = float(pricing.get("output_per_1k_tokens", -1))
    if input_rate < 0 or output_rate < 0:
        raise ValueError("pricing rates must be non-negative")
    result["status"] = "estimated_from_recorded_tokens"
    result["pricing"] = {
        "input_per_1k_tokens": input_rate,
        "output_per_1k_tokens": output_rate,
    }
    result["estimated_cost"] = (
        total["input_tokens"] / 1000 * input_rate
        + total["output_tokens"] / 1000 * output_rate
    )
    result["cost_scope"] = "recorded local artifacts; not a real provider invoice"
    return result


def _ledger_summary(path: Path, expected_ids: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Read one ledger and validate its stable call/attempt/result relationships."""
    errors: list[str] = []
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        logical = [dict(row) for row in db.execute("SELECT * FROM logical_calls ORDER BY logical_call_id")]
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY attempt_id")]
        results = [dict(row) for row in db.execute("SELECT * FROM analysis_results ORDER BY logical_call_id")]
    logical_ids = {row["logical_call_id"] for row in logical}
    if logical_ids != expected_ids:
        errors.append("ledger logical-call IDs do not match request artifacts")
    if len({row["logical_call_id"] for row in results}) != len(results):
        errors.append("duplicate analysis ownership exists")
    attempt_ids = {row["attempt_id"] for row in attempts}
    for row in results:
        if row["attempt_id"] not in attempt_ids:
            errors.append(f"analysis result points to missing attempt {row['attempt_id']}")
    for row in attempts:
        if row["response_ref"] and not (path.parent / row["response_ref"]).is_file():
            errors.append(f"missing response artifact for attempt {row['attempt_id']}")
    return {
        "logical_calls": len(logical),
        "attempts": len(attempts),
        "analysis_results": len(results),
        "unique_analysis_owners": len({row["logical_call_id"] for row in results}),
    }, errors


def _ledger_snapshot(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    """Capture restart-stable ledger rows without depending on timestamps."""
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return {
            table: [tuple(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("logical_calls", "attempts", "analysis_results")
        }


def _expected_call_ids(kind: str, run_dir: Path) -> set[str]:
    """Extract request call identities from diagnosis or repair artifacts."""
    if kind == "diagnosis":
        run = _load_object(run_dir / "run.json")
        return {
            str(row["request"]["logical_call_id"])
            for rows in run["methods"].values()
            for row in rows
        }
    return {str(row["call_id"]) for row in _load_rows(run_dir / "restricted" / "requests.json")}


def _usage_summary(path: Path) -> dict[str, int]:
    """Return recorded input/output/reasoning token totals for one ledger."""
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
            "COALESCE(SUM(reasoning_tokens),0), COALESCE(SUM(usage_status = 'unknown'),0) FROM attempts"
        ).fetchone()
    return {
        "attempts": int(row[0]),
        "input_tokens": int(row[1]),
        "output_tokens": int(row[2]),
        "reasoning_tokens": int(row[3]),
        "unknown_usage_attempts": int(row[4]),
    }


def _required_pairs(sd: float, half_width: float) -> int:
    """Use the normal approximation for a transparent planning baseline."""
    return max(1, math.ceil((Z_95 * sd / half_width) ** 2))


def _inflate_for_rate(required: int, rate: float | None) -> int | None:
    """Inflate evaluable pairs by the observed or conservative evaluable rate."""
    return None if rate is None or rate <= 0 else math.ceil(required / rate)


def _safe_check(name: str, check: Any) -> dict[str, Any]:
    """Turn malformed evidence into a named failed check instead of a false pass."""
    try:
        return check()
    except (OSError, TypeError, ValueError, KeyError, sqlite3.Error) as error:
        return {"status": "failed", "errors": [f"{name}: {type(error).__name__}: {error}"]}


def _evidence_status(value: Mapping[str, Any] | Path | None, name: str) -> dict[str, Any]:
    """Describe optional real-world evidence without inferring it from local artifacts."""
    if value is None:
        return {
            "status": "not_provided",
            "credential_status": "not_provided",
            "record_status": "not_provided",
        }
    if isinstance(value, Path) and not value.is_file():
        return {
            "status": "missing_record",
            "record": name,
            "credential_status": "not_provided",
            "record_status": "missing_record",
        }
    try:
        loaded = _load_object(value)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid_record",
            "record": name,
            "credential_status": "not_provided",
            "record_status": "invalid_record",
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "status": "provided",
        "record": name,
        "credential_status": loaded.get("credential_status", "not_provided"),
        "record_status": loaded.get("status", "unknown"),
    }


def _private_text_values(answer_key: Sequence[Mapping[str, Any]]) -> set[str]:
    """Collect nontrivial restricted text that must never appear in generation requests."""
    values: set[str] = set()
    for row in answer_key:
        for key in ("reference_answer", "reference_solution"):
            value = row.get(key)
            if isinstance(value, str) and len(value.strip()) >= 4:
                values.add(value)
        material = row.get("reference_material")
        if isinstance(material, Mapping):
            value = material.get("content")
            if isinstance(value, str) and len(value.strip()) >= 4:
                values.add(value)
    return values


def _nested_keys(value: Any) -> set[str]:
    """Return all object keys in a JSON-compatible request."""
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {key for child in value.values() for key in _nested_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _nested_keys(child)}
    return set()


def _number_or_none(value: Any) -> float | None:
    """Normalize a numeric JSON value while preserving missing rates."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _load_object(value: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    """Load a JSON object from a path or accept an in-memory object."""
    loaded = json.loads(value.read_text(encoding="utf-8")) if isinstance(value, Path) else value
    if not isinstance(loaded, Mapping):
        raise TypeError("expected a JSON object")
    return loaded


def _load_rows(value: Sequence[Mapping[str, Any]] | Path) -> list[Mapping[str, Any]]:
    """Load a JSON array of objects from a path or accept an in-memory sequence."""
    loaded = json.loads(value.read_text(encoding="utf-8")) if isinstance(value, Path) else value
    if not isinstance(loaded, list) or any(not isinstance(row, Mapping) for row in loaded):
        raise TypeError("expected a JSON array of objects")
    return loaded


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    """Load nonblank JSONL objects for offline repair recomputation."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError("JSONL rows must be objects")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one UTF-8 JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
