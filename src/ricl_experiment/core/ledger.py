"""SQLite ledger for logical calls, send attempts, responses, and budget use."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from ..config import canonical_json

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS logical_calls (
    logical_call_id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    owner_attempt_id TEXT,
    response_ref TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    logical_call_id TEXT NOT NULL REFERENCES logical_calls(logical_call_id),
    attempt_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    effective_parameters TEXT NOT NULL,
    reserved_output_tokens INTEGER NOT NULL,
    usage_status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    response_ref TEXT,
    error TEXT,
    duplicate_billing_risk INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    UNIQUE(logical_call_id, attempt_number)
);
CREATE TABLE IF NOT EXISTS analysis_results (
    logical_call_id TEXT PRIMARY KEY REFERENCES logical_calls(logical_call_id),
    attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
    response_ref TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Ledger:
    """Small transaction-oriented API over one run's SQLite database."""

    def __init__(self, path: Path, budget_tokens: int):
        """Open the run ledger and ensure its minimal schema exists."""
        self.path = path
        self.budget_tokens = budget_tokens
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the SQLite connection."""
        self.db.close()

    def ensure_call(self, call_id: str, sample_id: str, request_fingerprint: str) -> None:
        """Create the stable logical-call row once, leaving completed rows reusable."""
        self.db.execute(
            "INSERT OR IGNORE INTO logical_calls "
            "(logical_call_id, sample_id, request_fingerprint, status) VALUES (?, ?, ?, 'pending')",
            (call_id, sample_id, request_fingerprint),
        )
        row = self.db.execute(
            "SELECT request_fingerprint FROM logical_calls WHERE logical_call_id = ?", (call_id,)
        ).fetchone()
        if row["request_fingerprint"] != request_fingerprint:
            raise ValueError("logical call request fingerprint changed")

    def result(self, call_id: str) -> sqlite3.Row | None:
        """Return the already-owned analysis result, if one exists."""
        return self.db.execute(
            "SELECT attempt_id, response_ref FROM analysis_results WHERE logical_call_id = ?",
            (call_id,),
        ).fetchone()

    def call_status(self, call_id: str) -> str | None:
        """Return one logical call's state for resume retry decisions."""
        row = self.db.execute(
            "SELECT status FROM logical_calls WHERE logical_call_id = ?", (call_id,)
        ).fetchone()
        return row["status"] if row else None

    def recover_interrupted_attempts(self, call_id: str) -> None:
        """Turn a previous process's unfinished sends into auditable unknown attempts."""
        self.db.execute(
            "UPDATE attempts SET status = 'unknown', error = 'process ended before response was recorded' "
            "WHERE logical_call_id = ? AND status = 'pending'",
            (call_id,),
        )
        self.db.execute(
            "UPDATE logical_calls SET status = 'unknown' "
            "WHERE logical_call_id = ? AND owner_attempt_id IS NULL AND status != 'failed' "
            "AND EXISTS (SELECT 1 FROM attempts WHERE logical_call_id = ? AND status = 'unknown')",
            (call_id, call_id),
        )

    def unrecorded_attempts(self, call_id: str) -> list[sqlite3.Row]:
        """Return unresolved attempts that may have an uncommitted response file."""
        return self.db.execute(
            "SELECT attempt_id FROM attempts WHERE logical_call_id = ? "
            "AND status IN ('pending', 'unknown') AND response_ref IS NULL "
            "ORDER BY attempt_number",
            (call_id,),
        ).fetchall()

    def start_attempt(
        self,
        call_id: str,
        request_fingerprint: str,
        effective_parameters: dict[str, Any],
        reservation: int,
        max_attempts: int,
    ) -> str:
        """Persist an attempt and reserve its budget before the HTTP request is sent."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute(
                "SELECT attempt_number, status, usage_status FROM attempts "
                "WHERE logical_call_id = ? ORDER BY attempt_number",
                (call_id,),
            ).fetchall()
            if len(rows) >= max_attempts:
                raise ValueError(f"attempt limit ({max_attempts}) reached")
            budget = self.db.execute(
                "SELECT COALESCE(SUM(CASE WHEN usage_status = 'known' THEN output_tokens "
                "ELSE reserved_output_tokens END), 0) AS used FROM attempts"
            ).fetchone()["used"]
            if budget + reservation > self.budget_tokens:
                raise ValueError("output-token budget exhausted")
            attempt_id = uuid.uuid4().hex
            self.db.execute(
                "UPDATE attempts SET duplicate_billing_risk = 1 "
                "WHERE logical_call_id = ? AND status = 'unknown'",
                (call_id,),
            )
            self.db.execute(
                "INSERT INTO attempts (attempt_id, logical_call_id, attempt_number, status, "
                "request_fingerprint, effective_parameters, reserved_output_tokens, usage_status) "
                "VALUES (?, ?, ?, 'pending', ?, ?, ?, 'unknown')",
                (
                    attempt_id,
                    call_id,
                    len(rows) + 1,
                    request_fingerprint,
                    canonical_json(effective_parameters),
                    reservation,
                ),
            )
            self.db.execute(
                "UPDATE logical_calls SET status = 'pending' WHERE logical_call_id = ?", (call_id,)
            )
            self.db.commit()
            return attempt_id
        except Exception:
            self.db.rollback()
            raise

    def record_failure(
        self,
        attempt_id: str,
        status: str,
        error: str,
        usage: tuple[int, int] | None,
        response_ref: str | None = None,
    ) -> None:
        """Persist a failed or retryable send and keep unconfirmed usage reserved."""
        input_tokens, output_tokens = usage if usage is not None else (None, None)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE attempts SET status = ?, error = ?, usage_status = ?, input_tokens = ?, "
                "output_tokens = ?, response_ref = ?, completed_at = CURRENT_TIMESTAMP "
                "WHERE attempt_id = ?",
                (
                    status,
                    error[:1000],
                    "known" if usage is not None else "unknown",
                    input_tokens,
                    output_tokens,
                    response_ref,
                    attempt_id,
                ),
            )
            row = self.db.execute(
                "SELECT logical_call_id FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row:
                self.db.execute(
                    "UPDATE logical_calls SET status = ? WHERE logical_call_id = ? "
                    "AND owner_attempt_id IS NULL",
                    (status, row["logical_call_id"]),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def record_response(
        self,
        call_id: str,
        attempt_id: str,
        response_ref: str,
        usage: tuple[int, int] | None,
    ) -> None:
        """Record every received response and assign at most one analysis result."""
        input_tokens, output_tokens = usage if usage is not None else (None, None)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE attempts SET status = 'completed', usage_status = ?, input_tokens = ?, "
                "output_tokens = ?, response_ref = ?, error = NULL, "
                "completed_at = CURRENT_TIMESTAMP WHERE attempt_id = ?",
                (
                    "known" if usage is not None else "unknown",
                    input_tokens,
                    output_tokens,
                    response_ref,
                    attempt_id,
                ),
            )
            call = self.db.execute(
                "SELECT owner_attempt_id FROM logical_calls WHERE logical_call_id = ?", (call_id,)
            ).fetchone()
            if call is None:
                raise ValueError("logical call disappeared while recording response")
            if call["owner_attempt_id"] is None:
                self.db.execute(
                    "UPDATE logical_calls SET status = 'completed', owner_attempt_id = ?, response_ref = ? "
                    "WHERE logical_call_id = ?",
                    (attempt_id, response_ref, call_id),
                )
                self.db.execute(
                    "INSERT INTO analysis_results (logical_call_id, attempt_id, response_ref) "
                    "VALUES (?, ?, ?)",
                    (call_id, attempt_id, response_ref),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def close_run(self, status: str) -> None:
        """Persist the run's current completion state for audit consumers."""
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS run_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.db.execute(
            "INSERT INTO run_state (key, value) VALUES ('status', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (status,),
        )

    def usage_summary(self) -> dict[str, int]:
        """Summarize known use and still-reserved output tokens across attempts."""
        row = self.db.execute(
            "SELECT COALESCE(SUM(CASE WHEN usage_status = 'known' THEN output_tokens ELSE 0 END), 0) "
            "AS known, COALESCE(SUM(CASE WHEN usage_status = 'unknown' THEN reserved_output_tokens "
            "ELSE 0 END), 0) AS reserved, "
            "SUM(CASE WHEN usage_status = 'unknown' THEN 1 ELSE 0 END) AS unknown_attempts "
            "FROM attempts"
        ).fetchone()
        return {
            "known_output_tokens": row["known"],
            "reserved_output_tokens": row["reserved"],
            "unknown_usage_attempts": row["unknown_attempts"] or 0,
        }

