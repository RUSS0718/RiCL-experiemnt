"""Validate, adjudicate, and replay MS1 blind annotations without Provider requests."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from motivation_closeout import digest, read, write

from ricl_experiment.analysis.ms1_dependency import analyze_ms1_dependency
from ricl_experiment.motivation.ms1_dependency import import_ms1_annotations


def validate_rows(path, batch, reviewer, complete=True):
    rows = read(path)
    ids = {r["path_id"] for r in rows}
    if len(ids) != len(rows) or not ids <= batch.keys() or (complete and ids != batch.keys()):
        raise ValueError(f"wrong coverage in {path.name}")
    for row in rows:
        if row["reviewer_id"] != reviewer or row["reviewer_kind"] != "codex":
            raise ValueError(f"wrong provenance in {path.name}")
        prior = set()
        segments = batch[row["path_id"]]["segments"]
        for i, (node, segment) in enumerate(zip(row["nodes"], segments, strict=True)):
            sid = f"s{i:03}"
            if (
                node["step_id"] != sid
                or node["start"] != segment["start"]
                or node["end"] != segment["end"]
                or node["claim"] != segment["text"]
                or not set(node.get("dependency_ids", [])) <= prior
            ):
                raise ValueError(f"annotation changed the span or dependency order in {path.name}")
            if (
                node["validity_status"] in ("Valid", "Invalid")
                and not node["verification_evidence"].strip()
            ):
                raise ValueError(f"definitive verdict lacks a check in {path.name}")
            status = node.get("dependency_status", {})
            if isinstance(status, str) and status in ("confirmed", "excluded", "unknown"):
                # Older review files record a single status for all listed edges.
                node["dependency_status"] = dict.fromkeys(node.get("dependency_ids", []), status)
            prior.add(sid)
        anchor = row["first_error_step_id"]
        if (row["first_error_status"] == "confirmed" and anchor not in prior) or (
            row["first_error_status"] != "confirmed" and anchor is not None
        ):
            raise ValueError(f"bad first-error anchor in {path.name}")
    return rows


def votes(review):
    nodes = review["nodes"]
    groups = {n["step_id"]: n["equivalence_group_id"] or n["step_id"] for n in nodes}
    facts = {("anchor",): (review["first_error_status"], review["first_error_step_id"])}
    for node in nodes:
        sid = node["step_id"]
        facts["validity", sid] = node["validity_status"]
        facts["substantive", sid] = node["is_nontrivial"]
        facts["equivalence", sid] = tuple(sorted(k for k in groups if groups[k] == groups[sid]))
        for other in nodes[: int(sid[1:])]:
            dep = other["step_id"]
            facts["edge", sid, dep] = (
                node["dependency_status"].get(dep, "confirmed")
                if dep in node["dependency_ids"]
                else "excluded"
            )
    return facts


def consensus(row, third):
    a, b = row["independent_reviews"]
    x, y, z = votes(a), votes(b), votes(third) if third else {}
    out, unresolved, disagreement, agreement = {}, [], [], Counter()
    for key in x:
        agreement[f"{key[0]}_total"] += 1
        if key[0] == "edge" and (x[key], y[key]) != ("excluded", "excluded"):
            agreement["active_edge_total"] += 1
            agreement["active_edge_agree"] += x[key] == y[key]
        if x[key] == y[key]:
            agreement[f"{key[0]}_agree"] += 1
            out[key] = x[key]
        else:
            disagreement.append(key)
            if key in z and z[key] in (x[key], y[key]):
                out[key] = z[key]
            else:
                unresolved.append(key)
    status, anchor = out.get(("anchor",), ("unknown", None))
    nodes = []
    for node in a["nodes"]:
        sid = node["step_id"]
        members = out.get(("equivalence", sid))
        grouped = members and all(out.get(("equivalence", other)) == members for other in members)
        group = members[0] if grouped else sid
        validity = out.get(("validity", sid), "Unknown")
        evidence = [
            r["nodes"][int(sid[1:])]["verification_evidence"]
            for r in (a, b, third)
            if r and r["nodes"][int(sid[1:])]["validity_status"] == validity
        ]
        uncertain_claim = not grouped or ("substantive", sid) not in out
        if validity == "Valid" and uncertain_claim:
            validity = "Unknown"
        edges = {
            k[2]: out.get(k, "unknown")
            for k in x
            if k[:2] == ("edge", sid) and out.get(k, "unknown") != "excluded"
        }
        nodes.append(
            {
                **node,
                "validity_status": validity,
                "verification_evidence": " | ".join(e for e in evidence if e),
                "is_nontrivial": out.get(("substantive", sid), True),
                "equivalence_group_id": group,
                "dependency_ids": list(edges),
                "dependency_status": edges,
            }
        )
    return (
        {"first_error_status": status, "first_error_step_id": anchor, "nodes": nodes},
        disagreement,
        unresolved,
        agreement,
    )


def intervals(paths, seed):
    if not paths:
        return {}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paths), (20_000, len(paths)))
    intervals = {}
    for key in ("error_propagation_ratio", "reusable_claim_ratio"):
        value = np.array([[p[key][f] for f in ("numerator", "denominator")] for p in paths])
        counts = value[indices].sum(axis=1)
        counts = counts[counts[:, 1] > 0]
        intervals[key] = (
            np.quantile(counts[:, 0] / counts[:, 1], [0.025, 0.975]).tolist()
            if len(counts)
            else None
        )
    return intervals


def replay(frozen, batch_ids, third_root):
    review_root = frozen / "followup-review"
    blind = read(frozen / "ms1-blind.json")
    source = {r["blind_id"]: r for r in read(frozen / "restricted/ms1-source.json")}
    by_id = {r["path_id"]: r for r in blind["paths"]}
    if len(by_id) != 30 or set(source) != by_id.keys():
        raise ValueError("MS1 freeze is missing or inconsistent")
    all_rows, provenance, packets = [], {}, {}
    paired_agreement = Counter()
    reference_comparison = Counter()
    partial = {"pending": [], "paired": [], "third_records": 0, "unresolved_fields": 0}
    for i in batch_ids:
        batch = {r["path_id"]: r for r in read(review_root / f"ms1-blind-{i}.json")}
        names = [review_root / f"ms1-{i}-{side}.json" for side in ("a", "b")]
        if not all(p.is_file() for p in names):
            partial["pending"].append(i)
            continue
        if len(batch) != 10 or not batch.keys() <= by_id.keys():
            raise ValueError("the blind batch does not align with the frozen design")
        reviews = [
            r
            for p, side in zip(names, ("a", "b"), strict=True)
            for r in validate_rows(p, batch, f"codex-{i}-{side}")
        ]
        provenance.update({p.name: digest(p) for p in names})
        plan = {"selected_paths": list(batch.values()), "seed": blind["seed"]}
        pairs = import_ms1_annotations(plan, reviews)["paths"]
        conflicts = {r["path_id"] for r in pairs if consensus(r, None)[1]}
        packets[i] = [b for k, b in batch.items() if k in conflicts]
        p = third_root / f"ms1-third-{i}.json" if third_root else None
        adjudications = {}
        if p and p.is_file():
            third = validate_rows(
                p, {k: batch[k] for k in conflicts}, f"codex-third-{i}", complete=False
            )
            partial["third_records"] += len(third)
            provenance[p.name] = digest(p)
            normalized = import_ms1_annotations(plan, reviews, third)["paths"]
            adjudications = {
                r["path_id"]: r["third_review"] for r in normalized if r["third_review"]
            }
        partial["paired"].append(i)
        for row in pairs:
            k = row["path_id"]
            reconciled, differing, unresolved, agreement = consensus(row, adjudications.get(k))
            paired_agreement.update(agreement)
            status = reconciled["first_error_status"]
            truth = source[k]["error"]
            reference_comparison[f"adjudicated_{status}"] += 1
            if status == "confirmed":
                reference_comparison["confirmed_matches_source_index"] += (
                    int(reconciled["first_error_step_id"][1:]) == truth
                )
            if status == "none":
                reference_comparison["no_error_matches_source"] += truth == -1
            partial["unresolved_fields"] += len(unresolved)
            all_rows.append(
                {
                    "path_id": k,
                    "consensus": reconciled,
                    "disagreements": differing,
                    "unresolved": unresolved,
                    "third_reviewed": k in adjudications,
                }
            )
    if len({r["path_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("overlapping blind batches")
    analysis = analyze_ms1_dependency(all_rows)
    by_label = {}
    for label in ("source_error", "source_no_error"):
        subset = [
            r for r in all_rows if (source[r["path_id"]]["error"] >= 0) == (label == "source_error")
        ]
        a = analyze_ms1_dependency(subset)
        by_label[label] = {
            "summary": a["summary"],
            "denominators": a["denominators"],
            "descriptive_path_bootstrap95": intervals(a["paths"], blind["seed"]),
        }
    summary = {
        "schema": 1,
        "reviewer_kind": "codex",
        "review_count": 2 * len(all_rows),
        "coverage": partial,
        "agreement": paired_agreement,
        "comparison_to_public_labels": reference_comparison,
        "provenance": provenance,
        "by_source_label": by_label,
        "summary": analysis["summary"],
        "denominators": analysis["denominators"],
        "descriptive_path_bootstrap95": intervals(analysis["paths"], blind["seed"]),
    }
    return packets, summary, {"reconciled": all_rows, "analysis": analysis}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--third-root", type=Path)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3], choices=(1, 2, 3))
    a = p.parse_args()
    packets, summary, evidence = replay(a.frozen, a.batches, a.third_root)
    a.output.mkdir(parents=True, exist_ok=False)
    for i, packet in packets.items():
        write(a.output / f"ms1-third-packet-{i}.json", packet)
    write(a.output / "ms1-summary.json", summary)
    write(a.output / "ms1-per-path.json", evidence)
    print(summary["coverage"], {i: len(p) for i, p in packets.items()})


if __name__ == "__main__":
    main()
