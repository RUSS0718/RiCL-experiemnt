"""Replay the Motivation Study closeout from restricted local artifacts, without model calls."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath

import numpy as np
from motivation_eligibility import history, identity
from scipy.stats import binomtest


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.write("\n")


def ratio(count, total):
    ci = binomtest(count, total).proportion_ci() if total else None
    return {
        "n": count,
        "denominator": total,
        "value": count / total if total else None,
        "exact95": [ci.low, ci.high] if ci else None,
    }


def paired_interval(source_counts, seed):
    """A descriptive, source-stratified paired bootstrap on observed valid pairs."""
    rng = np.random.default_rng(seed)
    points = np.zeros(20_000)
    n = 0
    for row in source_counts.values():
        counts = np.array([row["counts"].get(k, 0) for k in ("loss", "tie", "win")])
        size = int(counts.sum())
        if size:
            points += rng.multinomial(size, counts / size, len(points)) @ [0, 0.5, 1]
            n += size
    return (np.quantile(points / n, [0.025, 0.975])).tolist() if n else None


def verify_frozen(root):
    manifest = root / "followup-review/frozen-input-hashes.json"
    hashes = read(manifest)
    for name, sha in hashes.items():
        path = root.joinpath(*PureWindowsPath(name).parts)
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"frozen input missing or changed: {name}")
    return {
        "files": len(hashes),
        "sqlite": sum(n.endswith(".sqlite3") for n in hashes),
        "manifest_sha256": digest(manifest),
    }


def first_batch(root):
    frozen_files = verify_frozen(root)
    observed = json.loads(
        subprocess.check_output(
            [sys.executable, str(root / "analyze_frozen_batch.py"), "--partial"], encoding="utf-8"
        )
    )
    if observed != read(root / "batch-results.json"):
        raise ValueError("first-batch replay differs from the frozen result")
    a, b, d = (observed[k] for k in ("ms2-a", "ms2-b", "ms3-a"))
    bad = {k: Counter() for k in a["source"]}
    for key, value in a["score_distribution"].items():
        source, side, score = key.split("|")
        if side == "right":
            bad[source][int(score)] += value
    high = {s: ratio(hist[2] + hist[3], sum(hist.values())) for s, hist in bad.items()}
    high["all"] = ratio(
        sum(r["n"] for r in high.values()), sum(r["denominator"] for r in high.values())
    )
    seed = read(root / "coverage.json")["seed"]
    ms3_missing = read(root / "followup-review/ms3-a-missingness.json")
    if ms3_missing["frozen_paired_coverage"] != [d["coverage"][k] for k in ("n", "denominator")]:
        raise ValueError("MS3-A coverage does not match the offline audit")
    return {
        "ms2_a": {
            "paired": a["overall"],
            "source": a["source"],
            "high_among_error_side": high,
            "requests": a["usage"],
        },
        "ms2_b": {
            "paired": b["overall"],
            "source": b["source"],
            "paired_concordance_bootstrap95": paired_interval(b["source"], seed),
            "bootstrap_groups": 20_000,
            "bootstrap_seed": seed,
            "requests": b["usage"],
        },
        "ms3_a": {
            "paired": d["coverage"],
            "strata": d["coverage_by_category"],
            "methods": ms3_missing["methods_by_source"],
            "request_disposition": ms3_missing["request_disposition"],
            "requests": d["usage"],
        },
        "total_with_probe": observed["total"],
        "verified_frozen_files": frozen_files,
    }


def exclusions(root, followup):
    """Retain explicit old IDs and canonical aliases, not mixed-count 'unique questions'."""
    seen, holds, sources = history()
    frozen = read(root / "coverage.json")["exclusion_sources"]
    normalized_sources = {PureWindowsPath(name).as_posix(): sha for name, sha in sources.items()}
    if any(
        normalized_sources.get(PureWindowsPath(name).as_posix()) != sha
        for name, sha in frozen.items()
    ):
        raise ValueError("an original historical exclusion source is missing or changed")
    slices = {}
    for name, key in (
        ("ms1-source", "gid"),
        ("ms2-a-pairs", "group"),
        ("ms2-b-pairs", "group"),
        ("ms3-a-labels", "gid"),
        ("ms3-b-qualification-queue", "gid"),
    ):
        path = root / f"restricted/{name}.json"
        rows = read(path)
        groups = {r[key] for r in rows}
        previously_allocated = holds.union(*(s["group_ids"] for s in slices.values()))
        if None in groups or len(groups) != len(rows) or groups & previously_allocated:
            raise ValueError(f"group overlap or missing identity in {name}")
        slices[name] = {"group_ids": sorted(groups), "source_sha256": digest(path)}
    if followup:
        replacements = read(followup)
        if digest(root / "coverage.json") != replacements["coverage_sha256"]:
            raise ValueError("MS3-B candidate continuation was not tied to this frozen batch")
        groups = {r["gid"] for r in replacements["rows"]}
        if groups & holds.union(*(s["group_ids"] for s in slices.values())) or any(
            identity(r["problem"]) != r["gid"] for r in replacements["rows"]
        ):
            raise ValueError("MS3-B replacement overlaps a prior group or has changed identity")
        slices["ms3-b-screened-backups"] = {
            "group_ids": sorted(groups),
            "source_sha256": digest(followup),
        }
    return {
        "historical_canonical_group_ids": sorted(seen),
        "historical_exclusion_keys": sorted(holds),
        "historical_source_sha256": dict(sorted(sources.items())),
        "revised_study_holds": slices,
        "all_exclusion_keys": sorted(holds.union(*(s["group_ids"] for s in slices.values()))),
        "note": "Retained qualification candidates are holds, not experimental observations. "
        "Re-run cross-dataset equivalence checks on a future Main Experiment candidate pool.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--ms3b-screened", type=Path)
    a = p.parse_args()
    evidence = first_batch(a.frozen)
    exclude = exclusions(a.frozen, a.ms3b_screened)
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / "first-batch-replay.json", evidence)
    write(a.output / "main-study-holds.json", exclude)
    print("first-batch replay matched; exclusion aliases:", len(exclude["all_exclusion_keys"]))


if __name__ == "__main__":
    main()
