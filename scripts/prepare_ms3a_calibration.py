"""Freeze a new, offline MS3-A calibration manifest disjoint from prior work."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ricl_experiment.config import fingerprint
from ricl_experiment.datasets.manifest import (
    _mark_near_duplicate_candidates,
    normalize_problem,
    problem_fingerprint,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motivation_eligibility import history, identity, load

SOURCES = ("pb:gsm8k", "pb:math", "pb:olympiadbench", "pb:omnimath")
SEED = 20261126
MANIFEST_ID = "ms3a-calibration-independent-20261126-v2"


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _historical_eligibility_inventory() -> tuple[
    set[str], dict[str, str], dict[str, str], list[str], dict[str, str]
]:
    """Include frozen eligibility cohorts omitted by the legacy history scanner."""
    groups: set[str] = set()
    questions: dict[str, str] = {}
    hashes: dict[str, str] = {}
    non_json_files: list[str] = []
    main_study_hold_hashes: dict[str, str] = {}
    for root in sorted(Path("artifacts").glob("motivation-eligibility-*")):
        for path in sorted(root.rglob("*.json")):
            raw = path.read_bytes()
            hashes[path.as_posix()] = hashlib.sha256(raw).hexdigest()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                non_json_files.append(path.as_posix())
                continue
            for row in _walk(payload):
                for key in ("question_group_id", "gid"):
                    value = row.get(key)
                    if isinstance(value, str) and value:
                        groups.add(value)
                value = row.get("group")
                if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                    groups.add(value)
                problem = row.get("problem")
                if isinstance(problem, str) and problem.strip():
                    questions[identity(problem)] = normalize_problem(problem)
    for path in sorted(Path("artifacts").rglob("main-study-holds.json")):
        raw = path.read_bytes()
        main_study_hold_hashes[path.as_posix()] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            continue
        for key in (
            "historical_canonical_group_ids",
            "historical_exclusion_keys",
            "all_exclusion_keys",
        ):
            groups.update(item for item in payload.get(key, []) if isinstance(item, str))
        revised = payload.get("revised_study_holds", {})
        if isinstance(revised, dict):
            for item in revised.values():
                if isinstance(item, dict):
                    groups.update(
                        value for value in item.get("group_ids", []) if isinstance(value, str)
                    )
    return groups, questions, hashes, non_json_files, main_study_hold_hashes


def prepare(output: Path) -> dict[str, Any]:
    """Select one source-stratified, pre-outcome-ranked group per ProcessBench source."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    rows, snapshots = load()
    seen, held, history_files = history()
    (
        frozen_groups,
        frozen_questions,
        frozen_hashes,
        non_json_files,
        main_study_hold_hashes,
    ) = _historical_eligibility_inventory()
    held.update(frozen_groups)
    held.update(seen)
    held.update(frozen_questions)

    questions = dict(seen)
    questions.update(frozen_questions)
    for row in rows:
        problem = row.get("problem")
        if isinstance(problem, str) and problem.strip():
            questions.setdefault(identity(problem), normalize_problem(problem))

    near_rows = [
        {
            "source_record_id": f"identity:{group_id}",
            "question_group_id": group_id,
            "problem": problem,
            "_identity_status": "unreviewed",
            "_identity_candidate_ids": [],
        }
        for group_id, problem in sorted(questions.items())
        if group_id
    ]
    _mark_near_duplicate_candidates(near_rows)
    near_groups = {row["question_group_id"] for row in near_rows if row["_identity_candidate_ids"]}

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions = Counter()
    for original in rows:
        source = original.get("source")
        if source not in SOURCES:
            continue
        row = dict(original)
        gid = identity(row["problem"])
        row["gid"] = gid
        if (
            gid in held
            or problem_fingerprint(row["problem"]) in held
            or f"problem-{problem_fingerprint(row['problem'])}" in held
        ):
            exclusions["historical_or_frozen"] += 1
            continue
        if gid in near_groups:
            exclusions["near_duplicate"] += 1
            continue
        steps, label = row.get("steps"), row.get("error")
        if (
            not isinstance(steps, list)
            or len(steps) < 2
            or not all(isinstance(step, str) and step.strip() for step in steps)
            or type(label) is not int
            or not -1 <= label < len(steps)
            or (label >= 0 and label >= len(steps) - 1)
        ):
            exclusions["invalid_shape"] += 1
            continue
        row["source_record_id"] = f"{source}:{row['rid']}"
        candidates[source].append(row)

    if any(not candidates[source] for source in SOURCES):
        counts = {source: len(candidates[source]) for source in SOURCES}
        raise ValueError(f"insufficient eligible candidates by source: {counts}")

    selected: list[dict[str, Any]] = []
    selected_originals: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    for source in SOURCES:
        ranked = sorted(
            candidates[source],
            key=lambda row: hashlib.sha256(
                f"{SEED}:ms3a-calibration:{source}:{row['gid']}:{row['rid']}".encode()
            ).hexdigest(),
        )
        row = next((item for item in ranked if item["gid"] not in selected_groups), None)
        if row is None:
            raise ValueError(f"no distinct eligible question group remains for {source}")
        selected_groups.add(row["gid"])
        selected_originals.append(row)
        sample_id = (
            "ms3a-cal-"
            + hashlib.sha256(f"{source}:{row['gid']}:{row['rid']}".encode()).hexdigest()[:20]
        )
        trajectory_id = (
            "cal-" + hashlib.sha256(f"trajectory:{source}:{row['rid']}".encode()).hexdigest()[:20]
        )
        selected.append(
            {
                "sample_id": sample_id,
                "trajectory_id": trajectory_id,
                "question_group_id": row["gid"],
                "source": "processbench",
                "source_stratum": source,
                "problem": row["problem"],
                "steps": row["steps"],
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "study": "ms3",
        "phase": "adjustment_validation",
        "sampling": {
            "seed": SEED,
            "rule": "sha256 rank within source after exact-history, frozen-group, and near-duplicate exclusions",
            "planned_per_source": 1,
            "sources": list(SOURCES),
            "snapshot_digests": snapshots,
            "historical_file_hashes": {**history_files, **frozen_hashes, **main_study_hold_hashes},
        },
        "samples": selected,
    }
    manifest["fingerprint"] = fingerprint(manifest)

    audit = {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_fingerprint": manifest["fingerprint"],
        "processbench_labels": [
            {
                "sample_id": chosen["sample_id"],
                "trajectory_id": chosen["trajectory_id"],
                "question_group_id": chosen["question_group_id"],
                "source_stratum": chosen["source_stratum"],
                "source_record_id": original["source_record_id"],
                "label": original["error"],
            }
            for chosen, original in zip(selected, selected_originals, strict=True)
        ],
        "provenance": {
            "selection": "one deterministic hash-ranked eligible group per ProcessBench source; no outcome-based resampling",
            "candidate_counts_by_source": {source: len(candidates[source]) for source in SOURCES},
            "excluded_rows": dict(exclusions),
            "historical_group_count": len(held),
            "main_study_holds_files": main_study_hold_hashes,
            "selected_label_by_source": {
                chosen["source_stratum"]: original["error"]
                for chosen, original in zip(selected, selected_originals, strict=True)
            },
            "near_duplicate_group_count": len(near_groups),
            "superseded_manifest": {
                "manifest_id": "ms3a-calibration-continuation-20261125",
                "status": "superseded; dispatch outcome unresolved",
                "reason": "selected from the frozen MS3-A confirmatory candidate pool; not an independent calibration set",
            },
            "labels_separated_from_runner_manifest": True,
            "unparsed_legacy_json_named_files": non_json_files,
            "previous_calibration_dispatch_record": (
                "artifacts/motivation-night-20261125/continuation-2326/dispatch-record.json"
            ),
            "previous_calibration_dispatch_status": "unknown; do not release its possible attempts",
        },
    }
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    restricted = output / "restricted"
    restricted.mkdir()
    (restricted / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "manifest_id": MANIFEST_ID,
        "fingerprint": manifest["fingerprint"],
        "selected": [
            {
                "source_stratum": row["source_stratum"],
                "sample_id": row["sample_id"],
                "question_group_id": row["question_group_id"],
                "trajectory_id": row["trajectory_id"],
            }
            for row in selected
        ],
        "candidate_counts_by_source": {source: len(candidates[source]) for source in SOURCES},
        "excluded_rows": dict(exclusions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
