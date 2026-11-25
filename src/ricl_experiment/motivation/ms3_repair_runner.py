"""MS3 paired repair execution and restricted artifact persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import RunConfig, fingerprint, validate_api_key
from ..core.ledger import Ledger
from ..core.model import ModelDispatchError, dispatch_request
from ..datasets.manifest import format_stratum, normalize_problem, problem_fingerprint
from ..runner import _run_lock
from .ms3_repair_protocol import (
    GROUPS,
    build_repair_request,
    choose_random_location,
    evaluate_repair_record,
    validate_repair_trajectory,
)


def run_ms3_repair(
    config: RunConfig,
    snapshot_path: Path,
    manifest_path: Path,
    audit_path: Path,
    answer_key_path: Path,
    output_dir: Path,
    *,
    seed: int | None = None,
    max_new_tokens: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Dispatch four isolated repair calls per eligible local ProcessBench sample."""
    manifest = _read_json_object(manifest_path)
    audit = _read_json_object(audit_path)
    answer_key_rows = _read_json_value(answer_key_path)
    answer_key = _index_rows(answer_key_rows, ("sample_id",))
    snapshot = _read_json_object(snapshot_path)
    snapshot_records = snapshot.get("records")
    source_records = _index_source_records(snapshot_records)
    if manifest.get("study") != "ms3" or snapshot.get("source") != "processbench":
        raise ValueError("MS3 repair requires an MS3 manifest and ProcessBench snapshot")
    _validate_snapshot_identity(manifest, snapshot)
    label_rows = _index_rows(
        audit.get("processbench_labels"), ("trajectory_source_id", "trajectory_id")
    )
    samples = _index_rows(manifest.get("samples"), ("sample_id",))
    repair_sample_ids = audit.get("study_pools", {}).get("repair", [])
    selected_seed = seed if seed is not None else int(manifest.get("sampling", {}).get("seed", 0))
    eligible, exclusions = _eligible_cases(
        repair_sample_ids, samples, answer_key, label_rows, source_records, snapshot,
        Path(snapshot_path).name, selected_seed
    )

    validate_api_key(config)
    if max_new_tokens is None:
        max_new_tokens = config.model.parameters["max_tokens"]
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if "max_tokens" not in config.model.capabilities.supported_parameters:
        raise ValueError("the configured model endpoint does not support max_tokens")
    parameters = {**config.model.parameters, "max_tokens": max_new_tokens}
    effective_config = _with_parameters(config, parameters)
    worst_case_reservation = (
        len(eligible) * len(GROUPS) * config.budget.max_attempts_per_call * max_new_tokens
    )
    if worst_case_reservation > config.budget.max_reserved_output_tokens:
        raise ValueError(
            "output-token budget cannot cover all four groups' configured attempt limits"
        )

    requests = {}
    for case in eligible:
        positions = (None, None, case["g2_position"], case["first_error_position"])
        for group, position in zip(GROUPS, positions, strict=True):
            requests[case["sample_id"], group] = build_repair_request(
                case["problem"], case["steps"], group, position, parameters, config.model.model,
            )

    output_dir = Path(output_dir)
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir():
        raise FileNotFoundError(f"MS3 repair run does not exist: {output_dir}")

    metadata = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "study": "ms3",
        "phase": manifest.get("phase"),
        "groups": list(GROUPS),
        "manifest_fingerprint": fingerprint(manifest),
        "answer_key_fingerprint": fingerprint(answer_key_rows),
        "seed": selected_seed,
        "max_new_tokens": max_new_tokens,
        "effective_parameters": parameters,
        "model": config.model.model,
        "endpoint": f"{config.model.base_url}/chat/completions",
        "tools": "disabled",
        "logical_calls_per_eligible_sample": len(GROUPS),
        "audit_fingerprint": fingerprint(audit),
        "snapshot_fingerprint": fingerprint(snapshot),
        "config_fingerprint": fingerprint(config.redacted()),
        "requests_fingerprint": fingerprint(list(requests.values())),
    }
    with _run_lock(output_dir):
        metadata_path = output_dir / "metadata.json"
        ledger_path = output_dir / "run.sqlite3"
        if resume:
            if not metadata_path.is_file() or not ledger_path.is_file():
                raise ValueError("MS3 repair run is missing its durable metadata or ledger")
            previous = _read_json_object(metadata_path)
            for field, value in metadata.items():
                if previous.get(field) != value:
                    raise ValueError(f"MS3 repair resume rejected: {field} changed")
            if previous.get("status") == "completed":
                return {
                    "assignments": _read_json_value(output_dir / "restricted" / "assignments.json"),
                    "evaluations": [
                        json.loads(line)
                        for line in (output_dir / "blind" / "evaluations.jsonl").read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ],
                    "usage": previous["usage"],
                    "status": "completed",
                }
        else:
            (output_dir / "blind").mkdir()
            (output_dir / "restricted").mkdir()
            _write_json(output_dir / "manifest.json", manifest)
            _write_json(output_dir / "eligibility.json", exclusions)
        _write_json(metadata_path, {**metadata, "status": "running"})

        assignments: list[dict[str, Any]] = []
        blind_records: list[dict[str, Any]] = []
        process_evidence: list[dict[str, Any]] = []
        request_audit: list[dict[str, Any]] = []
        run_status = "incomplete"
        ledger = Ledger(ledger_path, config.budget.max_reserved_output_tokens)
        try:
            for case in eligible:
                for group in GROUPS:
                    request = requests[case["sample_id"], group]
                    call_id = fingerprint({
                        "run_id": output_dir.name,
                        "sample_id": case["sample_id"],
                        "group": group,
                    })
                    evaluation_id = fingerprint({"call_id": call_id, "role": "blind-evaluation"})
                    request_audit.append({
                        "call_id": call_id,
                        "group": group,
                        "request": request,
                    })
                    _write_json(output_dir / "restricted" / "requests.json", request_audit)
                    assignment = {
                        "evaluation_id": evaluation_id,
                        "sample_id": case["sample_id"],
                        "trajectory_id": case["trajectory_id"],
                        "question_group_id": case["question_group_id"],
                        "group": group,
                        "call_id": call_id,
                        "g2_candidate_indices": case["g2_candidate_indices"],
                        "g2_position": case["g2_position"],
                        "first_error_position": case["first_error_position"],
                        "g3_position": case["first_error_position"],
                        "position_offset": case["position_offset"],
                        "trajectory_length": len(case["steps"]),
                        "stratum": case["stratum"],
                        "backbone": config.model.model,
                        "effective_parameters": parameters,
                        "max_new_tokens": max_new_tokens,
                        "tools_enabled": False,
                        "status": "pending",
                        "response_ref": None,
                    }
                    assignments.append(assignment)
                    process_evidence.append({
                        "evaluation_id": evaluation_id,
                        "first_error_position": case["first_error_position"],
                        "first_error_step": case["steps"][case["first_error_position"]],
                    })
                    try:
                        response_ref = dispatch_request(
                            effective_config,
                            ledger,
                            output_dir,
                            call_id,
                            f"{case['sample_id']}:{group}",
                            request,
                        )
                        candidate_output = _read_response(output_dir / response_ref)
                        status = "completed" if candidate_output.strip() else "no_evaluable_output"
                        missing_reason = None if candidate_output.strip() else "no_evaluable_output"
                    except ModelDispatchError as error:
                        response_ref = None
                        candidate_output = None
                        status = ledger.call_status(call_id) or "failed"
                        missing_reason = f"dispatch_{status}: {error}"
                    assignment.update({"status": status, "response_ref": response_ref})
                    assignment["attempts"] = _call_attempts(ledger, call_id)
                    blind_records.append({
                        "evaluation_id": evaluation_id,
                        "problem": case["problem"],
                        "original_steps": case["steps"],
                        "candidate_output": candidate_output,
                        "reference_material": case["reference_material"],
                        **evaluate_repair_record(
                            evaluation_id,
                            "missing",
                            "missing",
                            "missing",
                            missing_reason=missing_reason or "pending_independent_review",
                        ),
                    })
                    _write_json(output_dir / "restricted" / "assignments.json", assignments)
                    _write_jsonl(output_dir / "blind" / "evaluations.jsonl", blind_records)
                    _write_jsonl(output_dir / "restricted" / "first-error-evidence.jsonl", process_evidence)
            if all(row["response_ref"] is not None for row in assignments):
                run_status = "completed"
        finally:
            try:
                ledger.close_run(run_status)
                costs = _token_costs(ledger)
                _write_json(metadata_path, {**metadata, "status": run_status, "usage": costs})
            finally:
                ledger.close()
        _write_json(output_dir / "restricted" / "requests.json", request_audit)
        return {
            "assignments": assignments,
            "evaluations": blind_records,
            "usage": costs,
            "status": run_status,
        }


def _eligible_cases(
    sample_ids: Any,
    samples: Mapping[str, dict[str, Any]],
    answer_key: Mapping[str, dict[str, Any]],
    labels: Mapping[str, dict[str, Any]],
    source_records: Mapping[str, dict[str, Any]],
    snapshot: Mapping[str, Any],
    snapshot_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = []
    exclusions = []
    for sample_id_value in sample_ids:
        sample_id = str(sample_id_value)
        sample = samples.get(sample_id)
        key = answer_key.get(sample_id)
        if sample is None or key is None:
            raise ValueError(f"repair sample is missing manifest or restricted data: {sample_id}")
        source_id = str(sample.get("trajectory_source_id") or "")
        source_record = source_records.get(source_id)
        label = labels.get(source_id)
        if source_record is None or label is None:
            raise ValueError(f"repair sample cannot be joined to its ProcessBench source: {sample_id}")
        if key.get("expert_label") != label.get("label"):
            raise ValueError(f"restricted error labels disagree for sample: {sample_id}")
        first_error = label.get("label")
        steps = key.get("steps")
        source_steps = _preserved_steps(source_record.get("steps"))
        if not isinstance(steps, list) or any(not isinstance(step, str) for step in steps):
            raise ValueError(f"restricted sample is missing preserved source steps: {sample_id}")
        if steps != source_steps:
            raise ValueError(f"restricted steps do not match the pinned source record: {sample_id}")
        if source_record.get("label") != first_error:
            raise ValueError(f"audit label does not match the pinned source record: {sample_id}")
        native_id = next(
            (
                str(source_record[key])
                for key in ("source_record_id", "unique_id", "record_id", "id")
                if source_record.get(key) is not None and str(source_record[key]).strip()
            ),
            None,
        )
        if native_id is None:
            native_id = (
                f"file:{snapshot_name}:row:{source_record['_snapshot_index'] + 1}:"
                f"problem:{problem_fingerprint(normalize_problem(source_record.get('problem')))[:16]}"
            )
        expected_record_id = (
            f"{snapshot['source']}:{snapshot['revision']}:{snapshot['split']}:{native_id}"
        )
        if sample.get("source_record_id") != expected_record_id:
            raise ValueError(f"manifest record identity does not match the pinned source: {sample_id}")
        okay, reason = validate_repair_trajectory(steps, first_error)
        if not okay:
            exclusions.append({"sample_id": sample_id, "reason": reason})
            continue
        material = key.get("reference_material")
        if not _verified_reference(material):
            raise ValueError(f"repair sample lacks independently verified reference material: {sample_id}")
        raw_problem = source_record.get("problem")
        if not isinstance(raw_problem, str) or " ".join(raw_problem.split()) != sample.get("problem"):
            raise ValueError(f"source problem does not match frozen manifest sample: {sample_id}")
        trajectory_id = str(sample.get("trajectory_id") or source_id)
        random_position = choose_random_location(seed, trajectory_id, len(steps), first_error)
        subset = source_record.get(
            "source_subset", source_record.get("subset", source_record.get("source", snapshot.get("split")))
        )
        eligible.append({
            "sample_id": sample_id,
            "trajectory_id": trajectory_id,
            "question_group_id": str(sample.get("question_group_id") or ""),
            "problem": str(sample["problem"]),
            "steps": steps,
            "reference_material": material,
            "first_error_position": first_error,
            "g2_candidate_indices": [index for index in range(len(steps)) if index != first_error],
            "g2_position": random_position,
            "position_offset": random_position - first_error,
            "stratum": format_stratum(["processbench", subset, source_record.get("generator", "unknown"), "error"]),
        })
    return eligible, exclusions


def _index_source_records(value: Any) -> dict[str, dict[str, Any]]:
    """Index ProcessBench source records by their original trajectory identifier."""
    if not isinstance(value, list):
        raise TypeError("snapshot records must be an array of objects")
    result = {}
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise TypeError("snapshot records must be an array of objects")
        trajectory_id = row.get("trajectory_id")
        if trajectory_id is None:
            trajectory_id = row.get("id")
        if trajectory_id is None:
            continue
        result[str(trajectory_id)] = {**row, "_snapshot_index": index}
    return result


def _validate_snapshot_identity(manifest: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
    """Prevent running a manifest against another dataset revision or split."""
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise TypeError("MS3 manifest is missing pinned source metadata")
    expected = (snapshot.get("source"), snapshot.get("revision"), snapshot.get("split"))
    if not any((row.get("source"), row.get("revision"), row.get("split")) == expected for row in sources):
        raise ValueError("ProcessBench snapshot does not match the frozen manifest source revision")


def _verified_reference(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("kind") in {"answer", "solution"}
        and isinstance(value.get("content"), str)
        and bool(value["content"].strip())
        and isinstance(value.get("source"), str)
        and bool(value["source"].strip())
        and value.get("independently_verified") is True
    )


def _with_parameters(config: RunConfig, parameters: dict[str, Any]) -> RunConfig:
    model = config.model.model_copy(update={"parameters": parameters})
    return config.model_copy(update={"model": model})


def _preserved_steps(value: Any) -> list[str]:
    """Parse source steps without altering nonblank step text."""
    if isinstance(value, (list, tuple)):
        return [step for step in value if isinstance(step, str) and step.strip()]
    if isinstance(value, str):
        return [line for line in value.splitlines() if line.strip()]
    return []

def _index_rows(value: Any, keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON array of records")
    result = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise TypeError("record collection contains a non-object value")
        for key in keys:
            if row.get(key) is not None:
                result[str(row[key])] = dict(row)
    return result


def _read_json_value(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_response(path: Path) -> str:
    payload = _read_json_object(path)
    return str(payload["choices"][0]["message"]["content"])


def _call_attempts(ledger: Ledger, call_id: str) -> list[dict[str, Any]]:
    rows = ledger.db.execute(
        "SELECT attempt_number, status, input_tokens, output_tokens, reasoning_tokens, "
        "usage_status, reserved_output_tokens, response_ref, error "
        "FROM attempts WHERE logical_call_id = ? ORDER BY attempt_number",
        (call_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _token_costs(ledger: Ledger) -> dict[str, int]:
    row = ledger.db.execute(
        "SELECT COUNT(*) AS attempts, "
        "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
        "COALESCE(SUM(output_tokens), 0) AS output_tokens, "
        "COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens, "
        "SUM(CASE WHEN usage_status = 'unknown' THEN 1 ELSE 0 END) AS unknown_usage_attempts, "
        "COALESCE(SUM(CASE WHEN usage_status = 'unknown' THEN reserved_output_tokens ELSE 0 END), 0) "
        "AS reserved_output_tokens FROM attempts"
    ).fetchone()
    keys = (
        "attempts",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "unknown_usage_attempts",
        "reserved_output_tokens",
    )
    return {key: int(row[key] or 0) for key in keys}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )
    temporary.replace(path)
