"""Inventory and reserve disjoint public question groups for the revised Motivation Study.

Run from the repository root with the local ProcessBench snapshots and the pinned
PRM800K Phase 2 test file. Writes a new local artifact directory. No provider calls.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from ricl_experiment.datasets.manifest import (
    _mark_near_duplicate_candidates,
    load_snapshot,
    normalize_problem,
    problem_fingerprint,
)
from ricl_experiment.evaluation.answer import answers_equivalent, extract_final_answer
from ricl_experiment.motivation.ms1_dependency import build_blind_annotation_manifest
from ricl_experiment.motivation.ms2_scoring import build_blinded_rater_request
from ricl_experiment.motivation.ms3.protocol import METHODS, build_diagnosis

SEED = 20261125
PRM_COMMIT = "7ecc794703b2877f63226f2477a49b34f9b25163"
PRM_DIGEST = "6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad"
MODEL = "deepseek-v4.1-flash"
END = re.compile(
    r"\\boxed\s*\{|(?im:^\s*#*\s*(?:final\s+answer|answer)\s*(?::|$))"
    r"|(?i:\b(?:the (?:final )?answer is|hence the (?:final )?answer)\b)"
)
UNFINISHED = re.compile(
    r"(?i)\b(?:unable to|requires further|cannot determine|unable to determine)\b"
)


def dump(where, name, value):
    (where / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def identity(problem):
    return problem_fingerprint(normalize_problem(problem).casefold())


def history(*, source_paths: set[Path] | None = None):
    """Collect observed text and explicit group holds from local manifests and plans."""
    seen, holds, files = {}, set(), {}
    for directory in (Path("artifacts"), Path("data")):
        for path in sorted(directory.rglob("*.json")):
            if source_paths is not None and path not in source_paths:
                continue
            if "motivation-eligibility" in str(path) or not (
                "manifest" in path.name
                or path.name.endswith("plan.json")
                or "exclude" in path.name
                or "holdout" in path.name
            ):
                continue
            raw = path.read_bytes()
            obj = json.loads(raw)
            rows = obj.get("samples", []) if isinstance(obj, dict) else []
            exclusions = (
                obj.get("excluded_question_group_ids", [])
                if isinstance(obj, dict)
                else (
                    obj
                    if isinstance(obj, list) and ("exclude" in path.name or "holdout" in path.name)
                    else []
                )
            )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                question = row.get("problem") or row.get("prompt")
                if question and isinstance(question, str):
                    seen[identity(question)] = question
                    holds.update((problem_fingerprint(question), identity(question)))
                if row.get("question_group_id"):
                    holds.add(row["question_group_id"])
            holds.update(item for item in exclusions if isinstance(item, str))
            if rows or exclusions:
                files[str(path)] = hashlib.sha256(raw).hexdigest()
    return seen, holds, files


def aligned_ratings(steps, labeled):
    """Read only completions text-aligned with the original generated trajectory."""
    ratings = []
    for step, label in zip(steps, labeled, strict=False):
        matches = {
            x.get("rating")
            for x in label.get("completions", [])
            if normalize_problem(x.get("text")) == normalize_problem(step) and not x.get("flagged")
        }
        if len(matches) != 1 or next(iter(matches)) not in {-1, 0, 1}:
            break
        ratings.append(matches.pop())
    return ratings


def load():
    rows, sources = [], {}
    for path in sorted(Path("data/ms3-reference-review").glob("processbench-*.json")):
        snap = load_snapshot(path)
        sources[snap.split] = {"revision": snap.revision, "sha256": snap.snapshot_digest}
        for row in snap.records:
            steps = row["steps"]
            final = extract_final_answer(steps[-1]) if steps else None
            explicit_final = (
                final
                and not UNFINISHED.search(steps[-1])
                and not re.search(r"\\text(?:bf)?\{|[?…]", final)
            )
            rows.append(
                {
                    "source": f"pb:{snap.split}",
                    "rid": row["id"],
                    "problem": row["problem"],
                    "steps": steps,
                    "error": row["label"],
                    "generator": row["generator"],
                    "outcome": row["final_answer_correct"] if explicit_final else None,
                }
            )
    path = Path("data/prm800k-public/prm800k/data/phase2_test.jsonl")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PRM_DIGEST
    sources["prm:phase2_test"] = {"revision": PRM_COMMIT, "sha256": PRM_DIGEST}
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            row = json.loads(line)
            if row.get("is_quality_control_question") or row.get("is_initial_screening_question"):
                continue
            q, label = row["question"], row["label"]
            steps = q.get("pre_generated_steps") or []
            terminal = (
                re.search(r"(?im)^\s*#\s*Answer\s*(?:[:\n])\s*(\S[\s\S]*)$", steps[-1])
                if steps
                else None
            )
            answer = q.get("pre_generated_answer")
            gold = q.get("ground_truth_answer")
            outcome = None
            if (
                terminal
                and isinstance(answer, str)
                and isinstance(gold, str)
                and normalize_problem(terminal[1]) == normalize_problem(answer)
            ):
                outcome = answers_equivalent(answer, gold)
            ratings = aligned_ratings(steps, label["steps"])
            rows.append(
                {
                    "source": "prm:phase2_test",
                    "rid": f"phase2_test:{line_no}",
                    "problem": q["problem"],
                    "steps": steps,
                    "error": None,
                    "generator": None,
                    "outcome": outcome,
                    "ratings": ratings,
                }
            )
    return rows, sources


def eligible_pair(left, right, kind):
    a, b = left["steps"], right["steps"]
    if kind == "B":
        if left["outcome"] is not True or right["outcome"] is not False:
            return None
        k = min(8, min(len(a), len(b)) // 2)
    else:
        if left["source"].startswith("pb:"):
            if right["error"] < 0:
                return None
            k = right["error"] + 1
            if left["error"] >= 0 and left["error"] < k:
                return None
        else:
            if -1 not in right["ratings"]:
                return None
            k = right["ratings"].index(-1) + 1
            if len(left["ratings"]) < k or -1 in left["ratings"][:k]:
                return None
    if not (2 <= k < min(len(a), len(b))):
        return None
    one, two = ("\n\n".join(x[:k]) for x in (a, b))
    if normalize_problem(one) == normalize_problem(two) or END.search(one) or END.search(two):
        return None
    return k


def choose(pools, counts, groups_used):
    """Seed-rank candidates within each predeclared stratum without group reuse."""
    selected = []
    by_stratum = {}
    for key, count in counts.items():
        ranked = sorted(pools.get(key, []), key=lambda x: rank(key, x["gid"] + x["rid"]))
        rows = []
        for item in ranked:
            if item["gid"] not in groups_used:
                groups_used.add(item["gid"])
                rows.append(item)
                if len(rows) == count:
                    break
        by_stratum[key] = {"selected": len(rows), "planned": count}
        selected.extend(rows)
    return selected, by_stratum


def rank(key, value):
    return hashlib.sha256(f"{SEED}:{key}:{value}".encode()).hexdigest()


def pair_pools(grouped, counts):
    pools = {"A": defaultdict(list), "B": defaultdict(list)}
    for (source, gid), rows in sorted(grouped.items()):
        for kind in pools:
            match = None
            for a, b in product(rows, repeat=2):
                k = eligible_pair(a, b, kind)
                if k is not None:
                    key = rank(kind, a["rid"] + ":" + b["rid"])
                    if match is None or key < match[0]:
                        match = (key, a, b, k)
            if match:
                _, a, b, k = match
                counts[source]["pairs_" + kind] += 1
                pools[kind][source].append(
                    {"gid": gid, "rid": a["rid"] + ":" + b["rid"], "pair": (a, b), "cut": k}
                )
    return pools


def marshal_pairs(selected):
    """Separate blinded prefixes from public-source judgments and preserved suffix provenance."""
    blind, restricted = [], []
    for item in selected:
        k = item["cut"]
        pair = []
        for row in item["pair"]:
            bid = "prefix-" + rank("blind", row["source"] + ":" + row["rid"] + ":" + str(k))[:20]
            blind.append(
                {"id": bid, "problem": row["problem"], "checkpoint": "\n\n".join(row["steps"][:k])}
            )
            pair.append(
                {
                    "id": bid,
                    "source": row["source"],
                    "record": row["rid"],
                    "generator": row["generator"],
                    "outcome": row["outcome"],
                    "first_error": row["error"],
                    "ratings": row.get("ratings", [])[:k],
                }
            )
        restricted.append({"group": item["gid"], "cut": k, "left": pair[0], "right": pair[1]})
    blind.sort(key=lambda row: rank("order", row["id"]))
    return blind, restricted


def ms1_blind(rows):
    selected_paths = []
    for row in rows:
        bid = "path-" + rank("ms1-blind", row["rid"])[:16]
        row["blind_id"] = bid
        text = "\n\n".join(row["steps"])
        segments, start = [], 0
        for step in row["steps"]:
            segments.append(
                {
                    "kind": "source_step",
                    "status": "provided",
                    "start": start,
                    "end": start + len(step),
                    "text": step,
                }
            )
            start += len(step) + 2
        selected_paths.append(
            {
                "path_id": bid,
                "sample_id": bid,
                "candidate_id": bid,
                "question_group_id": row["gid"],
                "problem": row["problem"],
                "rollout_text": text,
                "segments": segments,
            }
        )
    return build_blind_annotation_manifest({"selected_paths": selected_paths})


def freeze_requests(out, a, b, three_a):
    """Persist the proposed 224 request bodies and a single development-mode probe."""
    parameters = {
        "max_tokens": 2048,
        "stream": False,
        "thinking": {"type": "enabled"},
        "response_format": {"type": "json_object"},
    }
    batches = {}
    for study, selected in (("MS2-A", a), ("MS2-B", b)):
        blind, _ = marshal_pairs(selected)
        bodies = {}
        for row in blind:
            body = build_blinded_rater_request(MODEL, row["problem"], row["checkpoint"], parameters)
            body["messages"][0]["content"] += (
                "\nRate only the displayed prefix. Do not solve the full problem or predict its"
                ' answer. Return JSON only, e.g. {"rating":2,"rationale":"brief explanation"}.'
            )
            bodies[row["id"]] = body
        batches[study] = bodies
    batches["MS3-A"] = {
        row["blind_id"] + ":" + method["id"]: build_diagnosis(method, row, MODEL, parameters)
        for row in three_a
        for method in METHODS
    }
    probe = build_diagnosis(
        METHODS[1],
        {
            "problem": "Compute the perimeter of a rectangle of length 7 and width 4.",
            "steps": [
                "The two adjacent side lengths sum to 7+4=11.",
                "There are two pairs of sides, so twice this sum is 24.",
                "The perimeter is therefore 24.",
            ],
        },
        MODEL,
        parameters,
    )
    oversize = {}
    for study, batch in batches.items():
        sizes = {k: len(json.dumps(v, ensure_ascii=False).encode()) + 256 for k, v in batch.items()}
        oversize[study] = {k: size for k, size in sizes.items() if size > 8192}
        dump(out, study.lower() + "-requests.json", batch)
    dump(
        out,
        "dispatch-proposal.json",
        {
            "authorized": False,
            "provider_base_url": "https://code.aigcpro.org/v1",
            "model": MODEL,
            "mode": parameters,
            "timeout_seconds": 135,
            "max_attempts_per_logical_call": 1,
            "auto_retries": False,
            "nominal_input_token_reservation": 8192,
            "request_byte_screen_with_overhead": 8192,
            "blocked_request_candidates": oversize,
            "proposed_billing_reservation_output_tokens_per_send": 3072,
            "proposed_total_max_usd": "1.40",
            "conditional_release_checks": [
                "Confirm current provider charge basis and enforceable spending bound.",
                "On the sole probe, require correct visible JSON, finish_reason stop and known usage.",
                "Require reasoning_tokens to be reported and no larger than completion_tokens.",
                "Pause on provider failure, missing or inconsistent usage, or an exceeded reservation.",
            ],
            "planned_study_requests": {study: len(batch) for study, batch in batches.items()},
            "probe": {
                "request": probe,
                "expected_first_error": 1,
                "maximum_sends": 1,
                "nominal_input_token_reservation": 2048,
                "nominal_output_token_reservation": 2048,
            },
            "global_coverage_required": 0.8,
            "historical_local_usd_rate_per_million": {"input": "0.30", "output": "1.20"},
            "provider_actual_rate_verified": False,
            "provider_actual_full_reconciliation": False,
            "ms3_b_deferred": "blind reference and pre-error conclusion qualification pending",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    assert not out.exists()
    rows, sources = load()
    seen, holds, historical = history()
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
    cnt = defaultdict(Counter)
    grouped = defaultdict(list)
    pb = defaultdict(list)
    for row in rows:
        source = row["source"]
        cnt[source]["rows"] += 1
        gid = row["gid"] = identity(row["problem"])
        if (
            gid in holds
            or problem_fingerprint(row["problem"]) in holds
            or ("problem-" + problem_fingerprint(row["problem"]) in holds)
        ):
            cnt[source]["historical_or_held"] += 1
            continue
        if gid in near_ids:
            cnt[source]["near_duplicate_held"] += 1
            continue
        steps = row["steps"]
        error = row["error"]
        if (
            not steps
            or not all(isinstance(s, str) and s.strip() for s in steps)
            or (
                source.startswith("pb:")
                and (len(steps) < 2 or type(error) is not int or not -1 <= error < len(steps))
            )
        ):
            cnt[source]["invalid_shape"] += 1
            continue
        cnt[source]["isolated_rows"] += 1
        cnt[source][f"terminal_{row['outcome']}"] += 1
        grouped[source, gid].append(row)
        if source.startswith("pb:"):
            pb[source].append(row)
    pools = pair_pools(grouped, cnt)
    used = set()
    subsets = sorted(pb)
    ms1_counts, ms3a_counts = {}, {}
    for i, source in enumerate(subsets):
        for status in ("error", "clean"):
            key = source + ":" + status
            ms1_counts[key] = 5 if status == "error" else (3 if i < 2 else 2)
            ms3a_counts[key] = 6 if status == "error" else 2
    single = {
        s + ":" + k: [
            r
            for r in rs
            if (r["error"] >= 0) == (k == "error")
            and (k == "clean" or r["error"] < len(r["steps"]) - 1)
        ]
        for s, rs in pb.items()
        for k in ("error", "clean")
    }
    # Reserve the independent MS1 pool before allocating paid-study candidates.
    one, one_counts = choose(single, ms1_counts, used)
    b, b_counts = choose(pools["B"], {s: 8 for s in subsets}, used)
    a, a_counts = choose(pools["A"], {**{s: 4 for s in subsets}, "prm:phase2_test": 16}, used)
    three_a, three_a_counts = choose(single, ms3a_counts, used)
    for row in three_a:
        row["blind_id"] = "trace-" + rank("ms3a", row["rid"])[:20]
    three_b_pools = defaultdict(list)
    for src, records in pb.items():
        hard = [
            r for r in records if len(r["steps"]) >= 8 and r["error"] >= 3 and r["gid"] not in used
        ]
        lengths = sorted(len(r["steps"]) for r in hard)
        median = lengths[len(lengths) // 2] if lengths else 0
        for row in hard:
            key = "upper" if len(row["steps"]) > median else "lower"
            row["length_stratum"] = key
            three_b_pools[key].append(row)
            cnt[src]["repair_shape_" + key] += 1
    three_b, three_b_counts = choose(three_b_pools, {"lower": 8, "upper": 8}, used)
    counts = {
        "MS1": one_counts,
        "MS2-A": a_counts,
        "MS2-B": b_counts,
        "MS3-A": three_a_counts,
        "MS3-B-candidate": three_b_counts,
    }
    restricted = out / "restricted"
    restricted.mkdir(parents=True)
    dump(out, "ms1-blind.json", ms1_blind(one))
    dump(restricted, "ms1-source.json", one)
    for label, items in (("ms2-a", a), ("ms2-b", b)):
        blind, key = marshal_pairs(items)
        dump(out, label + "-blind.json", blind)
        dump(restricted, label + "-pairs.json", key)
    dump(
        out,
        "ms3-a-blind.json",
        [{"id": r["blind_id"], "problem": r["problem"], "steps": r["steps"]} for r in three_a],
    )
    dump(restricted, "ms3-a-labels.json", three_a)
    dump(restricted, "ms3-b-qualification-queue.json", three_b)
    freeze_requests(out, a, b, three_a)
    dump(
        out,
        "coverage.json",
        {
            "seed": SEED,
            "sources": sources,
            "exclusion_sources": historical,
            "source_coverage": cnt,
            "plan_coverage": counts,
        },
    )
    print(json.dumps({"source_coverage": cnt, "plan_coverage": counts}, indent=2))


if __name__ == "__main__":
    main()
