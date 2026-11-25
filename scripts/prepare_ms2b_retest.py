"""Freeze the independent MS2-B historical-outcome retest without API dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ricl_experiment.config import fingerprint
from ricl_experiment.datasets.manifest import _mark_near_duplicate_candidates, normalize_problem
from ricl_experiment.motivation.ms2_scoring import build_blinded_rater_request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motivation_eligibility import MODEL, history, identity, load, marshal_pairs, pair_pools
from prepare_ms3a_calibration import _historical_eligibility_inventory

SOURCES = ("pb:gsm8k", "pb:math", "pb:olympiadbench", "pb:omnimath")
REFILL_ORDER = ("pb:math", "pb:olympiadbench", "pb:omnimath", "pb:gsm8k")
SEED = 20261126
MANIFEST_ID = "ms2b-independent-retest-20261126"
PARAMETERS = {
    "max_tokens": 2048,
    "stream": False,
    "thinking": {"type": "enabled"},
    "response_format": {"type": "json_object"},
}


def _near_duplicate_groups(questions: dict[str, str]) -> set[str]:
    rows = [
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
    _mark_near_duplicate_candidates(rows)
    return {row["question_group_id"] for row in rows if row["_identity_candidate_ids"]}


def prepare(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    rows, snapshots = load()
    seen, held, history_files = history()
    frozen_groups, frozen_questions, frozen_hashes, non_json, main_hold_hashes = (
        _historical_eligibility_inventory()
    )
    held.update(frozen_groups)
    held.update(seen)
    held.update(frozen_questions)

    questions = dict(seen)
    questions.update(frozen_questions)
    for row in rows:
        problem = row.get("problem")
        if isinstance(problem, str) and problem.strip():
            questions.setdefault(identity(problem), normalize_problem(problem))
    near_groups = _near_duplicate_groups(questions)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    excluded = Counter()
    for row in rows:
        source = row.get("source")
        if source not in SOURCES:
            continue
        gid = identity(row["problem"])
        if gid in held:
            excluded["historical_or_reserved"] += 1
            continue
        if gid in near_groups:
            excluded["near_duplicate"] += 1
            continue
        steps, label = row.get("steps"), row.get("error")
        if (
            not isinstance(steps, list)
            or len(steps) < 2
            or not all(isinstance(step, str) and step.strip() for step in steps)
            or type(label) is not int
            or not -1 <= label < len(steps)
        ):
            excluded["invalid_shape"] += 1
            continue
        row = dict(row)
        row["gid"] = gid
        grouped[source, gid].append(row)

    pools = pair_pools(grouped, defaultdict(Counter))["B"]
    pool_sizes = {source: len(pools.get(source, [])) for source in SOURCES}
    if sum(pool_sizes.values()) < 64:
        output.mkdir(parents=True)
        result = {
            "status": "blocked_insufficient_eligible_groups",
            "planned_groups": 64,
            "eligible_groups": sum(pool_sizes.values()),
            "candidate_counts_by_source": pool_sizes,
            "excluded_rows": dict(excluded),
            "study_snapshots": snapshots,
        }
        (output / "selection-summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    allocation = {source: min(16, pool_sizes[source]) for source in SOURCES}
    remaining = 64 - sum(allocation.values())
    while remaining:
        changed = False
        for source in REFILL_ORDER:
            if allocation[source] < pool_sizes[source]:
                allocation[source] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise AssertionError("64 eligible groups were counted but could not be allocated")

    selected = []
    selected_groups = set()
    for source in SOURCES:
        ranked = sorted(
            pools.get(source, []),
            key=lambda item: hashlib.sha256(
                f"{SEED}:ms2b-retest:{source}:{item['gid']}:{item['rid']}".encode()
            ).hexdigest(),
        )
        accepted = [item for item in ranked if item["gid"] not in selected_groups][
            : allocation[source]
        ]
        selected.extend(accepted)
        selected_groups.update(item["gid"] for item in accepted)
        if len(accepted) != allocation[source]:
            raise ValueError(f"could not allocate the frozen quota for {source}")
    if len(selected) != 64 or len(selected_groups) != 64:
        raise AssertionError("the frozen MS2-B selection must contain exactly 64 unique groups")

    blind, restricted_pairs = marshal_pairs(selected)
    pair_by_sample = {}
    for pair in restricted_pairs:
        for side in ("left", "right"):
            item = pair[side]
            pair_by_sample[item["id"]] = {
                "question_group_id": pair["group"],
                "source_stratum": item["source"],
            }
    samples = [
        {
            "sample_id": item["id"],
            **pair_by_sample[item["id"]],
            "source": "processbench",
            "problem": item["problem"],
            "checkpoint": item["checkpoint"],
        }
        for item in blind
    ]
    manifest = {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "study": "ms2",
        "phase": "adjustment_validation",
        "sampling": {
            "seed": SEED,
            "rule": "hash-rank eligible correct/incorrect trajectory pairs after frozen, historical, and near-duplicate exclusions",
            "planned_groups": 64,
            "initial_quota_per_source": 16,
            "refill_order": list(REFILL_ORDER),
            "selected_groups_by_source": allocation,
            "snapshot_digests": snapshots,
            "historical_file_hashes": {**history_files, **frozen_hashes, **main_hold_hashes},
        },
        "samples": samples,
    }
    manifest["fingerprint"] = fingerprint(manifest)

    requests = []
    for sample in samples:
        body = build_blinded_rater_request(
            MODEL, sample["problem"], sample["checkpoint"], PARAMETERS
        )
        body["messages"][0]["content"] += (
            "\nRate only the displayed prefix. Do not solve the full problem or predict its"
            ' answer. Return JSON only, e.g. {"rating":2,"rationale":"brief explanation"}.'
        )
        requests.append(
            {
                "sample_id": sample["sample_id"],
                "question_group_id": sample["question_group_id"],
                "source_stratum": sample["source_stratum"],
                "request_fingerprint": hashlib.sha256(
                    json.dumps(
                        body, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
        )
    plan = {
        "manifest_id": MANIFEST_ID,
        "manifest_fingerprint": manifest["fingerprint"],
        "model": MODEL,
        "parameters": PARAMETERS,
        "max_attempts_per_logical_call": 1,
        "logical_calls": len(requests),
        "requests": requests,
        "status": "prepared_not_dispatched",
        "provider_usage_status": "not_requested",
        "cost_usd": None,
        "cost_status": "unknown_not_zero",
    }
    audit = {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_fingerprint": manifest["fingerprint"],
        "pairs": restricted_pairs,
        "provenance": {
            "candidate_counts_by_source": pool_sizes,
            "selected_groups_by_source": allocation,
            "excluded_rows": dict(excluded),
            "historical_group_count": len(held),
            "near_duplicate_group_count": len(near_groups),
            "main_study_holds_file_hashes": main_hold_hashes,
            "unparsed_legacy_json_named_files": non_json,
            "labels_separated_from_blind_manifest": True,
        },
    }
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "request-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    restricted_dir = output / "restricted"
    restricted_dir.mkdir()
    (restricted_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "prepared_not_dispatched",
        "manifest_id": MANIFEST_ID,
        "manifest_fingerprint": manifest["fingerprint"],
        "groups": len(selected_groups),
        "logical_calls": len(requests),
        "selected_groups_by_source": allocation,
        "candidate_counts_by_source": pool_sizes,
        "excluded_rows": dict(excluded),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
