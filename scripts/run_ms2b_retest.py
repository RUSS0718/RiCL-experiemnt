"""Offline preflight and single-attempt dispatcher for the frozen MS2-B retest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ricl_experiment.config import (
    BudgetSettings,
    RunConfig,
    fingerprint,
    load_config,
    validate_api_key,
)
from ricl_experiment.core.ledger import Ledger
from ricl_experiment.core.model import dispatch_request, read_response, usage_out_of_bounds
from ricl_experiment.motivation.ms2_scoring import (
    build_blinded_rater_request,
    parse_llm_rater_response,
)

MODEL = "deepseek-v4.1-flash"
APPROVED_BASE_URL = "https://code.aigcpro.org/v1"
PARAMETERS = {
    "max_tokens": 2048,
    "stream": False,
    "thinking": {"type": "enabled"},
    "response_format": {"type": "json_object"},
}
PREFIX_ONLY_INSTRUCTION = (
    "\nRate only the displayed prefix. Do not solve the full problem or predict its"
    ' answer. Return JSON only, e.g. {"rating":2,"rationale":"brief explanation"}.'
)
MANIFEST_ID = "ms2b-independent-retest-20261126"
GROUP_COUNT = 64
CALL_COUNT = GROUP_COUNT * 2
MIN_PAIRS = 52
SHARED_ATTEMPT_CAP = 300
OLD_UNKNOWN_ATTEMPTS = 12
WINDOW_START = datetime.fromisoformat("2026-11-26T09:00:00+08:00")
WINDOW_END = datetime.fromisoformat("2026-11-26T17:00:00+08:00")
DEFAULT_FROZEN_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts/motivation-night-20261125/continuation-0011/ms2b-retest"
)
DISPATCH_GUARD_DIR = DEFAULT_FROZEN_DIR.parents[1] / "dispatch-guards"
PRIOR_RUN_DIR = DEFAULT_FROZEN_DIR.parents[1] / "continuation-20261126-0921/ms2b-live-dispatch/run"
PRIOR_RUN_ID = "8c521e110f894d3384fe2db8bc18db31"


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object used by the frozen preflight."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 for provenance in the dispatch record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_request_body(sample: dict[str, Any]) -> dict[str, Any]:
    """Rebuild exactly the frozen prefix-only MS2-B rating body."""
    body = build_blinded_rater_request(MODEL, sample["problem"], sample["checkpoint"], PARAMETERS)
    body["messages"][0]["content"] += PREFIX_ONLY_INSTRUCTION
    return body


def build_frozen_requests(frozen_dir: Path) -> list[dict[str, Any]]:
    """Validate frozen artifacts and rebuild each body against its stored fingerprint."""
    manifest_path = frozen_dir / "manifest.json"
    plan_path = frozen_dir / "request-plan.json"
    audit_path = frozen_dir / "restricted" / "audit.json"
    manifest = _read_json(manifest_path)
    plan = _read_json(plan_path)
    audit = _read_json(audit_path)

    manifest_core = dict(manifest)
    stored_manifest_fingerprint = manifest_core.pop("fingerprint", None)
    if stored_manifest_fingerprint != fingerprint(manifest_core):
        raise ValueError("manifest fingerprint mismatch")
    if manifest.get("manifest_id") != MANIFEST_ID:
        raise ValueError("unexpected frozen manifest id")
    if manifest.get("sampling", {}).get("planned_groups") != GROUP_COUNT:
        raise ValueError("frozen manifest must contain exactly 64 planned groups")
    if plan.get("manifest_id") != MANIFEST_ID:
        raise ValueError("request plan manifest id mismatch")
    if plan.get("manifest_fingerprint") != stored_manifest_fingerprint:
        raise ValueError("request plan manifest fingerprint mismatch")
    if plan.get("model") != MODEL or plan.get("parameters") != PARAMETERS:
        raise ValueError("model or effective parameters differ from the frozen retest")
    if plan.get("max_attempts_per_logical_call") != 1:
        raise ValueError("frozen plan must allow exactly one attempt per logical call")
    if plan.get("logical_calls") != CALL_COUNT:
        raise ValueError("frozen plan must contain exactly 128 logical calls")
    if plan.get("status") != "prepared_not_dispatched":
        raise ValueError("request plan is not in prepared_not_dispatched state")
    if (
        audit.get("manifest_id") != MANIFEST_ID
        or audit.get("manifest_fingerprint") != stored_manifest_fingerprint
    ):
        raise ValueError("restricted audit does not match the frozen manifest")

    samples = manifest.get("samples")
    planned_requests = plan.get("requests")
    pairs = audit.get("pairs")
    if not isinstance(samples, list) or len(samples) != CALL_COUNT:
        raise ValueError("frozen manifest must contain exactly 128 samples")
    if not isinstance(planned_requests, list) or len(planned_requests) != CALL_COUNT:
        raise ValueError("request plan must contain exactly 128 request fingerprints")
    if not isinstance(pairs, list) or len(pairs) != GROUP_COUNT:
        raise ValueError("restricted audit must contain exactly 64 pairs")

    samples_by_id = {
        sample.get("sample_id"): sample for sample in samples if isinstance(sample, dict)
    }
    if len(samples_by_id) != CALL_COUNT:
        raise ValueError("sample IDs must be unique")
    group_counts: dict[str, int] = {}
    for sample in samples:
        group = sample.get("question_group_id")
        if not isinstance(group, str) or not group:
            raise ValueError("sample is missing a question group")
        group_counts[group] = group_counts.get(group, 0) + 1
    if len(group_counts) != GROUP_COUNT or any(count != 2 for count in group_counts.values()):
        raise ValueError("manifest must contain exactly 64 paired groups")

    audited_ids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("group"), str):
            raise TypeError("restricted audit contains an invalid pair")
        sides = (pair.get("left"), pair.get("right"))
        if any(not isinstance(side, dict) or not isinstance(side.get("id"), str) for side in sides):
            raise ValueError("restricted audit pair is missing sample identities")
        side_ids = [side["id"] for side in sides]
        if side_ids[0] == side_ids[1] or any(sample_id in audited_ids for sample_id in side_ids):
            raise ValueError("restricted audit contains duplicate pair members")
        for sample_id in side_ids:
            sample = samples_by_id.get(sample_id)
            if sample is None or sample.get("question_group_id") != pair["group"]:
                raise ValueError("restricted audit pair identity differs from the manifest")
            audited_ids.add(sample_id)
    if audited_ids != set(samples_by_id):
        raise ValueError("restricted audit does not cover every frozen sample")

    rebuilt: list[dict[str, Any]] = []
    for sample, expected in zip(samples, planned_requests, strict=True):
        if not isinstance(expected, dict):
            raise TypeError("invalid planned request entry")
        identity = (
            sample.get("sample_id"),
            sample.get("question_group_id"),
            sample.get("source_stratum"),
        )
        planned_identity = (
            expected.get("sample_id"),
            expected.get("question_group_id"),
            expected.get("source_stratum"),
        )
        if identity != planned_identity:
            raise ValueError("request plan identity differs from the frozen manifest")
        body = build_request_body(sample)
        actual_fingerprint = fingerprint(body)
        if actual_fingerprint != expected.get("request_fingerprint"):
            raise ValueError(f"request fingerprint mismatch for {sample['sample_id']}")
        rebuilt.append(
            {
                "call_id": f"{MANIFEST_ID}:{sample['sample_id']}",
                "sample_id": sample["sample_id"],
                "question_group_id": sample["question_group_id"],
                "source_stratum": sample["source_stratum"],
                "request_fingerprint": actual_fingerprint,
                "body": body,
            }
        )
    return rebuilt


def request_reservation(known_ms3b_attempts: int, this_run_attempts: int) -> dict[str, int]:
    """Reserve this run and the old unknown sends against the shared 300-attempt cap."""
    if known_ms3b_attempts < 0 or this_run_attempts < 0:
        raise ValueError("attempt counts cannot be negative")
    total = known_ms3b_attempts + OLD_UNKNOWN_ATTEMPTS + this_run_attempts
    if total > SHARED_ATTEMPT_CAP:
        raise ValueError(
            f"request reservation {total} exceeds the shared {SHARED_ATTEMPT_CAP}-attempt cap"
        )
    return {
        "cap": SHARED_ATTEMPT_CAP,
        "known_ms3b_attempts": known_ms3b_attempts,
        "reserved_old_unknown_attempts": OLD_UNKNOWN_ATTEMPTS,
        "this_run_attempts": this_run_attempts,
        "total_reserved_attempts": total,
        "remaining_attempts": SHARED_ATTEMPT_CAP - total,
    }


def _completed_ms3b_attempts(completion_path: Path) -> tuple[dict[str, Any], int]:
    """Require a completed MS3-B record and count its persisted attempts read-only."""
    completion = _read_json(completion_path)
    if completion.get("status") != "completed":
        raise ValueError("current MS3-B completion record must say completed before dispatch")
    if completion.get("model") != MODEL or not str(completion.get("scope", "")).startswith("MS3-B"):
        raise ValueError("completion record is not for the current deepseek-v4.1-flash MS3-B run")
    run_dir_value = completion.get("run_dir")
    if not isinstance(run_dir_value, str) or not run_dir_value:
        raise ValueError("MS3-B completion record is missing run_dir")
    db_path = Path(run_dir_value) / "run.sqlite3"
    if not db_path.is_file():
        raise ValueError("MS3-B completion record points to a missing SQLite ledger")
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as db:
            row = db.execute("SELECT COUNT(*) FROM attempts").fetchone()
    except sqlite3.Error as error:
        raise ValueError("could not read the completed MS3-B attempt ledger") from error
    if row is None or not isinstance(row[0], int):
        raise ValueError("could not count completed MS3-B attempts")
    return completion, row[0]


def _write_json(path: Path, value: Any) -> None:
    """Atomically persist one run record before or between API attempts."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_requests(path: Path, requests: list[dict[str, Any]]) -> None:
    """Persist the exact request bodies and fingerprints before dispatch starts."""
    lines = [json.dumps(request, ensure_ascii=False, sort_keys=True) for request in requests]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frozen_hashes(frozen_dir: Path) -> dict[str, str]:
    return {
        name: _sha256(frozen_dir / name)
        for name in ("manifest.json", "request-plan.json", "restricted/audit.json")
    }


def _has_dispatch_time(config: RunConfig) -> bool:
    now = datetime.now(WINDOW_END.tzinfo)
    return WINDOW_START <= now < WINDOW_END - timedelta(seconds=config.model.timeout_seconds)


def _dispatch_context(
    config_path: Path, ms3b_completion_path: Path
) -> tuple[RunConfig, dict[str, int], dict[str, str]]:
    config = load_config(config_path)
    if config.model.model != MODEL:
        raise ValueError(f"config model must be exactly {MODEL}")
    if config.model.base_url.rstrip("/") != APPROVED_BASE_URL:
        raise ValueError("config provider must match the approved AIGCPro route")
    validate_api_key(config)
    completion, attempts = _completed_ms3b_attempts(ms3b_completion_path)
    config = config.model_copy(
        update={
            "budget": BudgetSettings(
                max_attempts_per_call=1,
                max_reserved_output_tokens=CALL_COUNT * PARAMETERS["max_tokens"],
            )
        }
    )
    if not _has_dispatch_time(config):
        raise ValueError("outside the authorized dispatch window")
    return (
        config,
        request_reservation(attempts, CALL_COUNT),
        {
            "path": str(ms3b_completion_path.resolve()),
            "sha256": _sha256(ms3b_completion_path),
            "status": completion["status"],
        },
    )


def _claim_dispatch(output_dir: Path, run_id: str) -> Path:
    guard = DISPATCH_GUARD_DIR / MANIFEST_ID
    owner = guard / "owner.json"
    guard.mkdir(parents=True, exist_ok=True)
    try:
        with owner.open("x", encoding="utf-8") as handle:
            json.dump({"run_id": run_id, "output_dir": str(output_dir)}, handle)
    except FileExistsError as error:
        raise ValueError(
            "the frozen manifest has already consumed its dispatch reservation"
        ) from error
    return guard


def run(
    frozen_dir: Path,
    output_dir: Path,
    *,
    dispatch: bool = False,
    config_path: Path | None = None,
    ms3b_completion_path: Path | None = None,
) -> dict[str, Any]:
    """Run offline preflight by default; dispatch only after explicit gated opt-in."""
    frozen_dir = frozen_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == frozen_dir or frozen_dir in output_dir.parents:
        raise ValueError("output directory must be separate from the frozen input directory")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    requests = build_frozen_requests(frozen_dir)
    hashes = _frozen_hashes(frozen_dir)
    reservation = None
    completion_metadata = None
    config = None
    run_id = uuid.uuid4().hex
    if dispatch:
        if config_path is None or ms3b_completion_path is None:
            raise ValueError("--dispatch requires --config and --ms3b-completion")
        config, reservation, completion_metadata = _dispatch_context(
            config_path, ms3b_completion_path
        )
        if (PRIOR_RUN_DIR / "dispatch-record.json").is_file():
            raise ValueError("the frozen manifest has already consumed its dispatch reservation")
        _claim_dispatch(output_dir, run_id)

    output_dir.mkdir(parents=True, exist_ok=False)
    restricted_dir = output_dir / "restricted"
    restricted_dir.mkdir()
    _write_requests(restricted_dir / "requests.jsonl", requests)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "prepared_not_dispatched",
        "scope": "MS2-B prefix rating only; no continuation or evidence extraction",
        "manifest_id": MANIFEST_ID,
        "created_artifact_hashes": hashes,
        "model": MODEL,
        "parameters": PARAMETERS,
        "logical_calls": CALL_COUNT,
        "max_attempts_per_logical_call": 1,
        "request_reservation": reservation,
        "config_fingerprint": fingerprint(config.redacted()) if config is not None else None,
        "ms3b_completion": completion_metadata,
        "calls": [
            {
                "call_id": request["call_id"],
                "sample_id": request["sample_id"],
                "question_group_id": request["question_group_id"],
                "request_fingerprint": request["request_fingerprint"],
                "status": "not_started",
                "response_ref": None,
                "error": None,
            }
            for request in requests
        ],
    }
    record_path = output_dir / "dispatch-record.json"
    _write_json(record_path, record)
    if not dispatch:
        record["status"] = "preflight_complete_not_dispatched"
        _write_json(record_path, record)
        return record

    assert config is not None
    return _dispatch(config, output_dir, requests, record, range(len(requests)))


def resume_dispatch(
    frozen_dir: Path,
    output_dir: Path,
    *,
    config_path: Path,
    ms3b_completion_path: Path,
) -> dict[str, Any]:
    """Continue once after a confirmed transport timeout using only unused slots."""
    output_dir = output_dir.resolve()
    requests = build_frozen_requests(frozen_dir)
    config, reservation, completion = _dispatch_context(config_path, ms3b_completion_path)
    record = _read_json(output_dir / "dispatch-record.json")
    if record.get("status") != "route_stopped" or "resumed_at" in record:
        raise ValueError("only the first stopped dispatch can be resumed")
    if any(
        record.get(key) != expected
        for key, expected in {
            "manifest_id": MANIFEST_ID,
            "created_artifact_hashes": _frozen_hashes(frozen_dir),
            "model": MODEL,
            "parameters": PARAMETERS,
            "logical_calls": CALL_COUNT,
            "max_attempts_per_logical_call": 1,
            "request_reservation": reservation,
            "config_fingerprint": fingerprint(config.redacted()),
            "ms3b_completion": completion,
        }.items()
    ):
        raise ValueError("existing dispatch record differs from the frozen plan")

    saved_path = output_dir / "restricted/requests.jsonl"
    saved_requests = [
        json.loads(line) for line in saved_path.read_text(encoding="utf-8").splitlines()
    ]
    if saved_requests != requests:
        raise ValueError("persisted dispatch requests differ from the frozen plan")

    stop_id = record["route_stop"]["call_id"]
    db_path = output_dir / "run.sqlite3"
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM attempts").fetchall()
    attempts = {row["logical_call_id"]: row for row in rows}
    stopped_attempt = attempts.get(stop_id)
    if (
        stopped_attempt is None
        or stopped_attempt["status"] != "unknown"
        or stopped_attempt["response_ref"] is not None
        or (stopped_attempt["error"] or "").split(";", 1)[0]
        not in {"ConnectTimeout", "ReadTimeout"}
    ):
        raise ValueError("the route stop is not a verified transport timeout")
    frozen_ids = {request["call_id"] for request in requests}
    if len(attempts) != len(rows) or not attempts.keys() <= frozen_ids:
        raise ValueError("ledger has duplicate or out-of-scope attempts")

    pending = []
    identity_keys = ("call_id", "sample_id", "question_group_id", "request_fingerprint")
    for index, (call, request) in enumerate(zip(record["calls"], requests, strict=True)):
        if any(call[key] != request[key] for key in identity_keys):
            raise ValueError("persisted call identity differs from the frozen plan")
        attempt = attempts.get(call["call_id"])
        if attempt is None:
            if call["status"] != "not_started":
                raise ValueError("dispatch record has an attempted slot missing from the ledger")
            pending.append(index)
            continue
        if (
            attempt["attempt_number"] != 1
            or attempt["request_fingerprint"] != call["request_fingerprint"]
            or json.loads(attempt["effective_parameters"]) != PARAMETERS
            or attempt["reserved_output_tokens"] != PARAMETERS["max_tokens"]
            or call["status"] != ("error" if call["call_id"] == stop_id else "completed")
            or call["response_ref"] != attempt["response_ref"]
        ):
            raise ValueError("ledger attempt differs from the frozen call or result state")

    guard = DISPATCH_GUARD_DIR / MANIFEST_ID
    owner = guard / "owner.json"
    if not owner.is_file() and output_dir == PRIOR_RUN_DIR.resolve():
        if record["run_id"] != PRIOR_RUN_ID:
            raise ValueError("unexpected original MS2-B run identity")
        try:
            _claim_dispatch(output_dir, PRIOR_RUN_ID)
        except ValueError:
            if not owner.is_file():
                raise
    if _read_json(owner) != {"run_id": record["run_id"], "output_dir": str(output_dir)}:
        raise ValueError("dispatch reservation belongs to a different run")
    (guard / "resume").mkdir(exist_ok=False)
    record["resumed_at"] = datetime.now(WINDOW_END.tzinfo).isoformat(timespec="seconds")
    return _dispatch(config, output_dir, requests, record, pending)


def _rating(payload: dict[str, Any]) -> tuple[str, int | None]:
    choice = payload["choices"][0]
    content = choice["message"]["content"]
    if choice.get("finish_reason") == "length":
        return "truncated", None
    if not content.strip():
        return "empty", None
    parsed = parse_llm_rater_response(content)
    return parsed["parse_status"], parsed["score"]


def _dispatch(
    config: RunConfig,
    output_dir: Path,
    requests: list[dict[str, Any]],
    record: dict[str, Any],
    pending: range | list[int],
) -> dict[str, Any]:
    record_path = output_dir / "dispatch-record.json"
    calls = record["calls"]
    missing_groups = {
        call["question_group_id"]
        for call in calls
        if call["status"] == "error"
        or (
            call["status"] == "completed"
            and _rating(read_response(output_dir, call["response_ref"]))[1] is None
        )
    }
    record["status"] = "dispatching"
    _write_json(record_path, record)
    ledger = Ledger(output_dir / "run.sqlite3", CALL_COUNT * PARAMETERS["max_tokens"])
    try:
        terminal = "completed"
        for index in pending:
            if GROUP_COUNT - len(missing_groups) < MIN_PAIRS:
                terminal = "coverage_unreachable"
                break
            if not _has_dispatch_time(config):
                terminal = "window_ended"
                break
            request = requests[index]
            call = calls[index]
            try:
                response_ref = dispatch_request(
                    config,
                    ledger,
                    output_dir,
                    request["call_id"],
                    request["sample_id"],
                    request["body"],
                    effective_parameters=PARAMETERS,
                )
            except Exception as error:  # noqa: BLE001 - every dispatch error stops this route
                call["status"] = "error"
                call["error"] = f"{type(error).__name__}: {error}"[:1000]
                terminal = "route_stopped"
                record["route_stop"] = {
                    "call_id": request["call_id"],
                    "call_status": ledger.call_status(request["call_id"]) or "not_started",
                }
                break
            call["status"] = "completed"
            call["response_ref"] = response_ref
            call["rating_status"], rating = _rating(read_response(output_dir, response_ref))
            attempt = ledger.call_audit(call["call_id"])["attempts"][0]
            if usage_out_of_bounds(attempt):
                call["rating_status"] = "usage_out_of_bounds"
                terminal = "usage_out_of_bounds"
                break
            if rating is None:
                missing_groups.add(call["question_group_id"])
            _write_json(record_path, record)
        record["status"] = terminal
        _write_json(record_path, record)
        ledger.close_run(terminal)
        return record
    except BaseException:
        record["status"] = "interrupted_unknown"
        _write_json(record_path, record)
        ledger.close_run("interrupted_unknown")
        raise
    finally:
        ledger.close()


def main() -> None:
    """Expose guarded offline and live modes as a command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dispatch", action="store_true", help="send after all offline gates pass")
    mode.add_argument("--resume-dispatch", action="store_true", help="resume unused frozen slots")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ms3b-completion", type=Path)
    args = parser.parse_args()
    try:
        if args.resume_dispatch:
            if args.config is None or args.ms3b_completion is None:
                parser.error("--resume-dispatch requires --config and --ms3b-completion")
            record = resume_dispatch(
                args.frozen_dir,
                args.output,
                config_path=args.config,
                ms3b_completion_path=args.ms3b_completion,
            )
        else:
            record = run(
                args.frozen_dir,
                args.output,
                dispatch=args.dispatch,
                config_path=args.config,
                ms3b_completion_path=args.ms3b_completion,
            )
    except (OSError, sqlite3.Error, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"run_id": record["run_id"], "status": record["status"]}))
    if record["status"] not in {"completed", "preflight_complete_not_dispatched"}:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
