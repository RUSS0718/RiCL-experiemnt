"""Replay the frozen MS2-B retest from the attempt ledger without model calls."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from motivation_closeout import paired_interval
from run_ms2b_retest import CALL_COUNT, GROUP_COUNT, MIN_PAIRS, _rating, build_frozen_requests

from ricl_experiment.core.model import read_response, usage_out_of_bounds


def summarize(frozen: Path, run_dir: Path) -> dict[str, Any]:
    requests = build_frozen_requests(frozen)
    record = json.loads((run_dir / "dispatch-record.json").read_bytes())
    manifest = json.loads((frozen / "manifest.json").read_bytes())
    pairs = json.loads((frozen / "restricted" / "audit.json").read_bytes())["pairs"]
    db_uri = f"{(run_dir / 'run.sqlite3').resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(db_uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM attempts").fetchall()
    attempts = {row["logical_call_id"]: row for row in rows}
    if len(attempts) != len(rows) or not attempts.keys() <= {r["call_id"] for r in requests}:
        raise ValueError("unexpected or duplicated attempts in the ledger")

    call_status: Counter[str] = Counter()
    scored: dict[str, int] = {}
    source_by_sample = {r["sample_id"]: r["source_stratum"] for r in requests}
    for request, call in zip(requests, record["calls"], strict=True):
        call_id, sample_id = request["call_id"], request["sample_id"]
        if (
            call["call_id"] != call_id
            or call["request_fingerprint"] != request["request_fingerprint"]
        ):
            raise ValueError("dispatch record differs from the frozen request plan")
        attempt = attempts.get(call_id)
        if attempt is None:
            if call["status"] != "not_started":
                raise ValueError("dispatch record reports an attempt absent from the ledger")
            call_status["not_started"] += 1
            continue
        if attempt["attempt_number"] != 1 or (
            attempt["request_fingerprint"] != request["request_fingerprint"]
            or attempt["response_ref"] != call["response_ref"]
        ):
            raise ValueError("attempt diverges from the frozen call or dispatch record")
        if attempt["status"] != "completed":
            call_status[attempt["status"]] += 1
            continue
        if usage_out_of_bounds(attempt):
            call_status["usage_out_of_bounds"] += 1
            continue
        status, score = _rating(read_response(run_dir, attempt["response_ref"]))
        call_status[status] += 1
        if score is not None:
            scored[sample_id] = score

    by_source = {}
    outcomes: dict[str, Counter[str]] = {}
    for pair in pairs:
        correct, incorrect = pair["left"], pair["right"]
        if correct["outcome"] is not True or incorrect["outcome"] is not False:
            raise ValueError("restricted pair has invalid historical outcome labels")
        left, right = correct["id"], incorrect["id"]
        source = source_by_sample[left]
        if source_by_sample[right] != source:
            raise ValueError("paired samples disagree on their source")
        coverage = by_source.setdefault(source, {"planned": 0, "scored": 0})
        coverage["planned"] += 1
        if left not in scored or right not in scored:
            continue
        coverage["scored"] += 1
        diff = scored[left] - scored[right]
        outcomes.setdefault(source, Counter())[
            "win" if diff > 0 else "loss" if diff < 0 else "tie"
        ] += 1

    paired = sum(source["scored"] for source in by_source.values())
    seed = manifest["sampling"]["seed"]
    effect = None
    if paired >= MIN_PAIRS:
        total = sum(outcomes.values(), Counter())
        effect = {
            "counts": dict(total),
            "by_source": {source: dict(counts) for source, counts in outcomes.items()},
            "concordance": (total["win"] + total["tie"] / 2) / paired,
            "source_stratified_paired_bootstrap95": paired_interval(
                {source: {"counts": counts} for source, counts in outcomes.items()}, seed
            ),
        }

    return {
        "dispatch_status": record["status"],
        "run_id": record["run_id"],
        "manifest_fingerprint": manifest["fingerprint"],
        "sampling_seed": seed,
        "planned_calls": CALL_COUNT,
        "actual_sends": len(rows),
        "call_disposition": dict(call_status),
        "planned_pairs": GROUP_COUNT,
        "valid_pairs": paired,
        "minimum_pairs": MIN_PAIRS,
        "coverage_by_source": by_source,
        "effect": effect,
        "usage": {
            "known_input_tokens": sum(row["input_tokens"] or 0 for row in rows),
            "known_output_tokens": sum(row["output_tokens"] or 0 for row in rows),
            "reported_reasoning_tokens": sum(row["reasoning_tokens"] or 0 for row in rows),
            "unknown_input_or_output_calls": sum(row["usage_status"] == "unknown" for row in rows),
            "unknown_reasoning_calls": sum(row["reasoning_tokens"] is None for row in rows),
        },
        "actual_cost_usd": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.frozen_dir, args.run_dir)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
