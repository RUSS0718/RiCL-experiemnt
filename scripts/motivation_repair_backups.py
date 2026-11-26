"""Freeze the next two MS3-B backup lists under the original strata and seed."""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from motivation_closeout import digest, read, write
from motivation_eligibility import (
    SEED,
    _mark_near_duplicate_candidates,
    choose,
    history,
    identity,
    load,
    normalize_problem,
    problem_fingerprint,
)


def freeze(frozen, replay, per_layer):
    rows, versions = load()
    coverage = read(frozen / "coverage.json")
    if coverage["seed"] != SEED or coverage["sources"] != versions:
        raise ValueError("the data snapshot or ranking seed differs from the original freeze")
    seen, holds, _ = history()
    questions = {**seen, **{identity(r["problem"]): r["problem"] for r in rows}}
    near = [
        {
            "source_record_id": gid,
            "question_group_id": gid,
            "problem": normalize_problem(q),
            "_identity_status": "unreviewed",
            "_identity_candidate_ids": [],
        }
        for gid, q in sorted(questions.items())
    ]
    _mark_near_duplicate_candidates(near)
    near_ids = {r["question_group_id"] for r in near if r["_identity_candidate_ids"]}
    slices = read(replay)["revised_study_holds"]
    used = {
        g
        for name, item in slices.items()
        if name != "ms3-b-qualification-queue"
        for g in item["group_ids"]
    }
    hard = defaultdict(list)
    for row in rows:
        steps, error, src = row["steps"], row["error"], row["source"]
        if not (
            src.startswith("pb:")
            and len(steps) >= 8
            and type(error) is int
            and 3 <= error < len(steps)
            and all(isinstance(s, str) and s.strip() for s in steps)
        ):
            continue
        group = identity(row["problem"])
        fingerprint = problem_fingerprint(row["problem"])
        if (
            group in used
            or group in near_ids
            or {group, fingerprint, f"problem-{fingerprint}"} & holds
        ):
            continue
        hard[src].append({**row, "gid": group})
    strata = defaultdict(list)
    medians = {}
    for source, group in hard.items():
        lengths = sorted(len(r["steps"]) for r in group)
        medians[source] = lengths[len(lengths) // 2]
        for row in group:
            layer = "upper" if len(row["steps"]) > medians[source] else "lower"
            strata[layer].append({**row, "length_stratum": layer})
    previous, _ = choose(strata, {"lower": 8, "upper": 8}, used)
    frozen_ids = [
        (r["gid"], r["rid"], r["length_stratum"])
        for r in read(frozen / "restricted/ms3-b-qualification-queue.json")
    ]
    if [(r["gid"], r["rid"], r["length_stratum"]) for r in previous] != frozen_ids:
        raise ValueError("cannot reproduce the original 8+8 order; do not substitute")
    selected, counts = choose(strata, {"lower": per_layer, "upper": per_layer}, used)
    return {
        "coverage_sha256": digest(frozen / "coverage.json"),
        "seed": SEED,
        "medians": medians,
        "counts": counts,
        "rows": selected,
    }


def qualify(frozen, reserve_path, review_root):
    reserve = read(reserve_path)
    original = read(frozen / "restricted/ms3-b-qualification-queue.json")
    sources = [frozen / f"followup-review/ms3b-review-{batch}.json" for batch in (1, 2)]
    sources += sorted(review_root.glob("ms3b-backup-review-*.json"))
    answers = {}
    for path in sources:
        for row in read(path):
            if row["gid"] in answers:
                raise ValueError("duplicated qualification review")
            answers[row["gid"]] = row
    available = {r["gid"]: r for r in original + reserve["rows"]}
    if not {r["gid"] for r in original} <= answers.keys() <= available.keys():
        raise ValueError(
            "qualification does not cover the original queue, or added unreserved groups"
        )
    for group, review in answers.items():
        row = available[group]
        if any(row[k] != review[k] for k in ("rid", "source", "length_stratum")):
            raise ValueError("qualification row does not match frozen candidate")
        if review["verdict"] not in ("qualified", "ineligible", "pending"):
            raise ValueError("invalid eligibility disposition")
        if review["verdict"] == "qualified":
            claim = review.get("pre_error_claim")
            if isinstance(claim, str):
                match = re.match(r"step\s+(\d+)\b", claim, re.IGNORECASE)
                claim_step = int(match[1]) if match else None
            else:
                claim_step = (claim or {}).get("step_index")
            first_error_confirmed = review.get("first_error_status") == "confirmed"
            if review.get("first_error_status") == "verified":
                match = re.search(r"\bindex\s+(\d+)\b", review["first_error_note"], re.IGNORECASE)
                first_error_confirmed = bool(match and int(match[1]) == row["error"])
            reference = review.get("reference_evidence")
            if (
                not reference
                and review.get("provenance")
                and isinstance(review.get("evidence"), dict)
            ):
                # The lower tail submitted its exact certificate under the shorter field name.
                reference = review["evidence"]
            if not (
                review.get("reference_status") == "verified"
                and reference
                and first_error_confirmed
                and type(claim_step) is int
                and 0 <= claim_step < row["error"]
            ):
                raise ValueError(
                    "eligible candidate has no verified reference, first error or prior claim"
                )
    filled = {}
    for layer in ("lower", "upper"):
        base = [r["gid"] for r in original if r["length_stratum"] == layer]
        ranked = [r["gid"] for r in reserve["rows"] if r["length_stratum"] == layer]
        screened = [g for g in ranked if g in answers]
        if screened != ranked[: len(screened)]:
            raise ValueError("screened backups skip a higher-ranked reserve in the same layer")
        accepted = [g for g in base + screened if answers[g]["verdict"] == "qualified"][:8]
        filled[layer] = {
            "original": Counter(answers[g]["verdict"] for g in base),
            "backup": Counter(answers[g]["verdict"] for g in screened),
            "eligible_group_ids": accepted,
            "shortage": 8 - len(accepted),
        }
    selected = {g for layer in filled.values() for g in layer["eligible_group_ids"]}
    return {
        "coverage_sha256": reserve["coverage_sha256"],
        "seed": reserve["seed"],
        "rows": [r for r in reserve["rows"] if r["gid"] in answers],
        "layers": filled,
        "eligible_groups": len(selected),
        "eligible_by_source": Counter(available[g]["source"] for g in sorted(selected)),
        "eligibility_gate_passed": all(not v["shortage"] for v in filled.values()),
        "reviews_sha256": {p.name: digest(p) for p in sources},
        "decisions": list(answers.values()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", required=True, type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initial-holds", type=Path)
    mode.add_argument("--reserve", type=Path)
    p.add_argument("--review-root", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--per-layer", default=8, type=int, choices=range(1, 9))
    a = p.parse_args()
    if a.reserve:
        if a.review_root is None:
            p.error("--reserve requires --review-root")
        result = qualify(a.frozen, a.reserve, a.review_root)
        write(a.output, result)
        print(
            {
                k: {x: y for x, y in v.items() if x != "eligible_group_ids"}
                for k, v in result["layers"].items()
            }
        )
    else:
        selection = freeze(a.frozen, a.initial_holds, a.per_layer)
        write(a.output, selection)
        print(selection["counts"])


if __name__ == "__main__":
    main()
