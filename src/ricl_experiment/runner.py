"""Single-process run lifecycle, artifact snapshots, and cross-process run locking."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .config import (
    RunConfig,
    SampleManifest,
    fingerprint,
    load_config,
    load_manifest,
    validate_api_key,
    validate_resume_inputs,
)
from .core.ledger import Ledger
from .core.model import ModelDispatchError, dispatch_sample


class RunLockError(RuntimeError):
    """Raised when another scheduler already owns a run directory."""


def start_run(
    study: str, phase: str, config_path: Path, manifest_path: Path, artifacts_root: Path
) -> str:
    """Create a frozen single-sample run and dispatch it through its SQLite ledger."""
    config, manifest = _load_inputs(config_path, manifest_path)
    validate_api_key(config)
    run_id = uuid.uuid4().hex
    run_dir = _run_dir(artifacts_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = _metadata(run_id, study, phase, config, manifest)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "config.json", config.redacted())
    _write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
    try:
        with _run_lock(run_dir):
            _execute(run_dir, metadata, config, manifest)
    except ModelDispatchError as error:
        raise ModelDispatchError(f"{error}; run_id={run_id}") from error
    return run_id


def resume_run(
    run_id: str, config_path: Path, manifest_path: Path, artifacts_root: Path
) -> str:
    """Resume an explicitly named run only when all frozen fingerprints still match."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id contains invalid path characters")
    config, manifest = _load_inputs(config_path, manifest_path)
    validate_api_key(config)
    run_dir = _run_dir(artifacts_root, run_id)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_resume_inputs(config, manifest, metadata)
    if metadata.get("run_id") != run_id:
        raise ValueError("resume rejected: run metadata identity does not match the requested run_id")
    with _run_lock(run_dir):
        _execute(run_dir, metadata, config, manifest)
    return run_id


def _load_inputs(config_path: Path, manifest_path: Path) -> tuple[RunConfig, SampleManifest]:
    """Load both frozen inputs before creating a run or allowing a network send."""
    return load_config(config_path), load_manifest(manifest_path)


def _metadata(
    run_id: str, study: str, phase: str, config: RunConfig, manifest: SampleManifest
) -> dict[str, object]:
    """Build a redacted manifest of run identities, fingerprints, and effective settings."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "study": study,
        "phase": phase,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "created",
        "config_fingerprint": fingerprint(config.redacted()),
        "manifest_fingerprint": fingerprint(manifest.model_dump(mode="json")),
        "prompt_fingerprint": fingerprint([sample.prompt for sample in manifest.samples]),
        "experiment_id": manifest.manifest_id,
        "dataset_version": manifest.dataset_version,
        "model": config.model.model,
        "endpoint": config.model.base_url,
        "effective_parameters": config.model.parameters,
        "redacted_config": config.redacted(),
    }


def _execute(
    run_dir: Path, metadata: dict[str, object], config: RunConfig, manifest: SampleManifest
) -> None:
    """Run the manifest's one stable logical call and update auditable run state."""
    sample = manifest.samples[0]
    call_id = fingerprint(
        {
            "run_id": metadata["run_id"],
            "study": metadata["study"],
            "phase": metadata["phase"],
            "sample_id": sample.sample_id,
            "sample_index": 0,
        }
    )
    request = {
        "model": config.model.model,
        "messages": [{"role": "user", "content": sample.prompt}],
        **config.model.parameters,
    }
    metadata["request_summary"] = {
        "logical_call_id": call_id,
        "sample_id": sample.sample_id,
        "roles": ["user"],
        "request_fingerprint": fingerprint(request),
        "effective_parameters": config.model.parameters,
    }
    _write_json(run_dir / "metadata.json", metadata)
    ledger = Ledger(run_dir / "run.sqlite3", config.budget.max_reserved_output_tokens)
    try:
        dispatch_sample(config, ledger, run_dir, call_id, sample)
        status = "completed"
    except ModelDispatchError:
        status = "incomplete"
        ledger.close_run(status)
        metadata["status"] = status
        metadata["usage"] = ledger.usage_summary()
        _write_json(run_dir / "metadata.json", metadata)
        raise
    finally:
        if ledger.db:
            try:
                ledger.close_run(status if "status" in locals() else "incomplete")
            finally:
                ledger.close()
    metadata["status"] = status
    with sqlite3.connect(run_dir / "run.sqlite3") as db:
        usage = db.execute(
            "SELECT COALESCE(SUM(CASE WHEN usage_status = 'known' THEN output_tokens ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN usage_status = 'unknown' THEN reserved_output_tokens ELSE 0 END), 0), "
            "SUM(CASE WHEN usage_status = 'unknown' THEN 1 ELSE 0 END) FROM attempts"
        ).fetchone()
    metadata["usage"] = {
        "known_output_tokens": usage[0],
        "reserved_output_tokens": usage[1],
        "unknown_usage_attempts": usage[2] or 0,
    }
    _write_json(run_dir / "metadata.json", metadata)


def _run_dir(artifacts_root: Path, run_id: str) -> Path:
    """Resolve the canonical artifact directory for one run identifier."""
    return artifacts_root.resolve() / "motivation" / run_id


def _write_json(path: Path, value: object) -> None:
    """Atomically write UTF-8 JSON metadata without partially written snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    """Hold a nonblocking OS file lock so a run has one scheduler process."""
    lock_path = run_dir / ".scheduler.lock"
    try:
        with lock_path.open("a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                if lock_file.read(1) == b"":
                    lock_file.seek(0)
                    lock_file.write(b"0")
                    lock_file.flush()
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise RunLockError("another scheduler already owns this run_id") from error
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    raise RunLockError("another scheduler already owns this run_id") from error
                try:
                    yield
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
    except RunLockError:
        raise
    except OSError as error:
        raise RunLockError("another scheduler already owns this run_id") from error
