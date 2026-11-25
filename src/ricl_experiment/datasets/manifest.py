"""Frozen-source sample identity, qualification, stratified sampling, and manifests."""

from __future__ import annotations

import hashlib
import json
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config import fingerprint

PINNED_SOURCES: dict[str, tuple[str, str]] = {
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
    "omni_math": ("KbsdJames/Omni-MATH", "40ba231d8f16e29ecd40e6407e2c8640145a8f62"),
    "processbench": ("Qwen/ProcessBench", "3bdcd5371ed567559a78f559c01c13a6deee7604"),
}
MANIFEST_SCHEMA_VERSION = 1
SAMPLING_RULE_VERSION = "sample-manifest-v1"
ANSWER_RULE_VERSION = "closed-answer-v1"
IDENTITY_RULE_VERSION = "normalized-exact-near-review-0.97-v1"


@dataclass(frozen=True)
class SourceSnapshot:
    """A local, already-materialized snapshot of one pinned dataset split."""

    source: str
    revision: str
    split: str
    records: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    snapshot_name: str | None = None
    snapshot_digest: str | None = None

    def __post_init__(self) -> None:
        if self.source not in PINNED_SOURCES:
            raise ValueError(f"unsupported source: {self.source}")
        expected_revision = PINNED_SOURCES[self.source][1]
        if self.revision != expected_revision:
            raise ValueError(
                f"{self.source} must use pinned revision {expected_revision}, not {self.revision}"
            )
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("snapshot split must not be empty")
        if any(not isinstance(record, Mapping) for record in self.records):
            raise ValueError("snapshot records must be objects")
        object.__setattr__(self, "records", tuple(dict(record) for record in self.records))
        if self.snapshot_digest is not None and (
            len(self.snapshot_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.snapshot_digest)
        ):
            raise ValueError("snapshot digest must be a SHA-256 hex digest")


@dataclass(frozen=True)
class SamplingPlan:
    """Frozen sampling inputs shared by prepare and later study consumers."""

    study: str
    phase: str
    seed: int
    quotas: Mapping[str, int]
    excluded_question_group_ids: tuple[str, ...] = ()
    rule_version: str = SAMPLING_RULE_VERSION

    def __post_init__(self) -> None:
        if self.study not in {"ms2", "ms3"}:
            raise ValueError("study must be ms2 or ms3")
        if self.phase not in {"pilot", "adjustment_validation", "confirmatory"}:
            raise ValueError("phase must be pilot, adjustment_validation, or confirmatory")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("sampling seed must be a non-negative integer")
        if not isinstance(self.quotas, Mapping) or not self.quotas:
            raise ValueError("at least one positive sampling quota is required")
        if any(
            not isinstance(key, str) or not key.strip() or type(value) is not int or value < 1
            for key, value in self.quotas.items()
        ):
            raise ValueError("sampling quotas must map non-empty strata to positive integers")
        if self.rule_version != SAMPLING_RULE_VERSION:
            raise ValueError(f"unsupported sampling rule version: {self.rule_version}")
        if any(not isinstance(group, str) or not group.strip() for group in self.excluded_question_group_ids):
            raise ValueError("excluded question-group IDs must be non-empty strings")
        object.__setattr__(self, "quotas", dict(self.quotas))
        object.__setattr__(
            self,
            "excluded_question_group_ids",
            tuple(sorted(set(self.excluded_question_group_ids))),
        )


@dataclass(frozen=True)
class PreparedManifest:
    """Blind manifest and separated restricted audit/reference artifacts."""

    manifest: dict[str, Any]
    audit: dict[str, Any]
    answer_key: list[dict[str, Any]]


def format_stratum(parts: Sequence[Any]) -> str:
    """Format an unambiguous canonical stratum key used in plans and manifests."""
    return json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))


def normalize_problem(problem: Any) -> str:
    """Normalize only whitespace for exact identity matching; preserve math content."""
    if not isinstance(problem, str):
        return ""
    return " ".join(problem.split()).strip()


def problem_fingerprint(problem: Any) -> str:
    """Return the stable fingerprint of a normalized problem statement."""
    return hashlib.sha256(normalize_problem(problem).encode("utf-8")).hexdigest()


def load_snapshot(path: Path) -> SourceSnapshot:
    """Load one JSON snapshot document and pin its exact local bytes by digest."""
    snapshot_path = Path(path)
    raw = snapshot_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("snapshot file must contain one JSON object")
    records = payload.get("records")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise TypeError("snapshot records must be an array of objects")
    return SourceSnapshot(
        source=str(payload.get("source", "")),
        revision=str(payload.get("revision", "")),
        split=str(payload.get("split", "")),
        records=tuple(records),
        snapshot_name=snapshot_path.name,
        snapshot_digest=hashlib.sha256(raw).hexdigest(),
    )


def load_sampling_plan(path: Path) -> SamplingPlan:
    """Load a strict JSON sampling plan for the public prepare CLI."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("sampling plan must contain one JSON object")
    allowed = {
        "study",
        "phase",
        "seed",
        "quotas",
        "excluded_question_group_ids",
        "rule_version",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown sampling plan fields: {', '.join(sorted(unknown))}")
    return SamplingPlan(**payload)


def write_prepared_manifest(prepared: PreparedManifest, output_dir: Path) -> dict[str, Path]:
    """Atomically write public manifest and separate restricted JSON sidecars."""
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}-") as temp:
        staging = Path(temp)
        (staging / "manifest.json").write_text(
            json.dumps(prepared.manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        restricted = staging / "restricted"
        restricted.mkdir()
        for name, content in (("audit", prepared.audit), ("answer-key", prepared.answer_key)):
            (restricted / f"{name}.json").write_text(
                json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        staging.rename(destination)
    return {
        "manifest": destination / "manifest.json",
        "audit": destination / "restricted" / "audit.json",
        "answer-key": destination / "restricted" / "answer-key.json",
    }


def write_overlap_report(
    left_manifest_path: Path, right_manifest_path: Path, output_path: Path
) -> Path:
    """Persist same-stage MS2/MS3 original-question overlap for audit."""
    left = json.loads(Path(left_manifest_path).read_text(encoding="utf-8"))
    right = json.loads(Path(right_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(left, dict) or not isinstance(left.get("samples"), list):
        raise TypeError("left manifest must contain a samples array")
    if not isinstance(right, dict) or not isinstance(right.get("samples"), list):
        raise TypeError("right manifest must contain a samples array")
    if {left.get("study"), right.get("study")} != {"ms2", "ms3"}:
        raise ValueError("overlap report requires one MS2 and one MS3 manifest")
    if left.get("phase") != right.get("phase"):
        raise ValueError("overlap report requires manifests from the same phase")
    overlap = phase_overlap(left["samples"], right["samples"])
    report = {
        "schema_version": 1,
        "left_manifest_fingerprint": left.get("fingerprint"),
        "right_manifest_fingerprint": right.get("fingerprint"),
        "overlap_count": len(overlap),
        "question_group_ids": overlap,
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"output file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def export_question_group_exclusions(
    confirmation_manifest_path: Path, output_path: Path
) -> Path:
    """Export the unique original-question groups in a frozen manifest as a JSON list."""
    payload = json.loads(Path(confirmation_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise TypeError("confirmation manifest must contain a samples array")
    if payload.get("study") != "ms2" or payload.get("phase") != "confirmatory":
        raise ValueError("exclusion list must come from an MS2 confirmatory manifest")
    groups = build_question_group_exclusion_list(payload["samples"])
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"output file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(groups, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def prepare_manifest(snapshot: SourceSnapshot, plan: SamplingPlan) -> PreparedManifest:
    """Prepare one deterministic manifest from a pinned local snapshot."""
    return prepare_sample_manifest((snapshot,), plan)


def prepare_sample_manifest(
    snapshots: Iterable[SourceSnapshot],
    plan: SamplingPlan,
) -> PreparedManifest:
    """Qualify, stratify, sample, and freeze records from local pinned snapshots."""
    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("at least one source snapshot is required")
    allowed_sources = {"math500", "omni_math"} if plan.study == "ms2" else {"processbench"}
    if any(snapshot.source not in allowed_sources for snapshot in snapshots):
        raise ValueError(f"{plan.study} cannot sample the supplied source type")
    source_keys = [(snapshot.source, snapshot.split) for snapshot in snapshots]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("source/split snapshots must be unique in one manifest")

    all_rows: list[dict[str, Any]] = []
    processbench_labels: list[dict[str, Any]] = []
    for snapshot in snapshots:
        rows, labels = _materialize_snapshot(snapshot, plan)
        all_rows.extend(rows)
        processbench_labels.extend(labels)
    _mark_near_duplicate_candidates(all_rows)

    source_record_ids = [row["source_record_id"] for row in all_rows]
    if len(source_record_ids) != len(set(source_record_ids)):
        raise ValueError("source record IDs must be unique within a manifest")

    identity_by_source_id = {row["source_record_id"]: row for row in all_rows}
    identity_by_native_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_by_record_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row.get("trajectory_source_id") is not None:
            identity_by_native_id[str(row["trajectory_source_id"])].append(row)
        if row.get("_source_native_id") is not None:
            identity_by_record_id[str(row["_source_native_id"])].append(row)
    identity_reviews = []
    identity_review_groups: set[str] = set()
    for row in all_rows:
        candidate_ids = row.get("_identity_candidate_ids", [])
        if row["_identity_status"] not in {"pending", "pending_review", "unresolved", "needs_review"} and not candidate_ids:
            continue
        candidates_by_id = {
            value: (
                [identity_by_source_id[value]]
                if value in identity_by_source_id
                else identity_by_native_id.get(value, []) + identity_by_record_id.get(value, [])
            )
            for value in candidate_ids
        }
        resolved_rows = {
            candidate["source_record_id"]: candidate
            for matches in candidates_by_id.values()
            for candidate in matches
        }
        resolved_ids = set(resolved_rows)
        identity_review_groups.add(row["question_group_id"])
        identity_review_groups.update(
            candidate["question_group_id"] for candidate in resolved_rows.values()
        )
        review = {
            "source_record_id": row["source_record_id"],
            "question_group_id": row["question_group_id"],
            "candidate_source_record_ids": sorted(resolved_ids),
            "reason": row.get("_identity_review_reason") or "identity pending review",
        }
        unresolved_ids = sorted(value for value, matches in candidates_by_id.items() if not matches)
        if unresolved_ids:
            review["unresolved_candidate_ids"] = unresolved_ids
        identity_reviews.append(review)

    exclusions: list[dict[str, Any]] = []
    qualified_rows: list[dict[str, Any]] = []
    for row in all_rows:
        reasons = _qualification_reasons(row["source"], row["_restricted_record"], row)
        if row["question_group_id"] in identity_review_groups and "identity_pending_review" not in reasons:
            reasons.insert(0, "identity_pending_review")
        if reasons:
            reason = reasons[0]
            detail = row.get("_identity_review_reason") or "; ".join(reasons)
            exclusions.append(_exclusion(row, reason, detail, reasons))
        else:
            qualified_rows.append(row)

    qualified_by_stratum: dict[str, int] = defaultdict(int)
    for row in qualified_rows:
        qualified_by_stratum[row["_sampling_stratum"]] += 1

    all_rows = qualified_rows

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["question_group_id"] in plan.excluded_question_group_ids:
            exclusions.append(_exclusion(row, "question_group_excluded", "frozen group exclusion list"))
            continue
        grouped[row["_sampling_stratum"]].append(row)

    selected: list[dict[str, Any]] = []
    for stratum, quota in sorted(plan.quotas.items()):
        candidates = sorted(grouped.pop(stratum, []), key=lambda row: row["source_record_id"])
        if len(candidates) < quota:
            raise ValueError(
                f"quota {quota} for stratum {stratum!r} exceeds qualified pool {len(candidates)}"
            )
        rng = random.Random(_stable_seed(plan.seed, stratum))
        chosen_ids = {row["source_record_id"] for row in rng.sample(candidates, quota)}
        for row in candidates:
            if row["source_record_id"] in chosen_ids:
                selected.append(row)
            else:
                exclusions.append(_exclusion(row, "quota_not_selected", "qualified but outside quota"))

    for stratum, candidates in sorted(grouped.items()):
        for row in candidates:
            exclusions.append(
                _exclusion(row, "stratum_not_requested", f"no quota was requested for {stratum!r}")
            )

    selected.sort(key=lambda row: (row["source"], row["source_record_id"]))
    public_samples = [_without_private_record(row) for row in selected]
    answer_key = [
        _answer_key_row(row, row["_restricted_record"])
        for row in selected
    ]
    source_metadata = [
        {
            "source": snapshot.source,
            "dataset": PINNED_SOURCES[snapshot.source][0],
            "revision": snapshot.revision,
            "split": snapshot.split,
            "snapshot_fingerprint": _snapshot_fingerprint(snapshot),
        }
        for snapshot in sorted(snapshots, key=lambda item: (item.source, item.split))
    ]
    strata = {
        stratum: {
            "requested": plan.quotas.get(stratum, 0),
            "qualified": count,
            "selected": sum(row["_sampling_stratum"] == stratum for row in selected),
            "excluded": count - sum(row["_sampling_stratum"] == stratum for row in selected),
        }
        for stratum in sorted(set(qualified_by_stratum) | set(plan.quotas))
        for count in [qualified_by_stratum.get(stratum, 0)]
    }
    audit_body: dict[str, Any] = {
        "schema_version": 1,
        "rule_version": plan.rule_version,
        "identity_rule_version": IDENTITY_RULE_VERSION,
        "sources": source_metadata,
        "qualified_count": len(all_rows),
        "qualified_question_group_count": len(
            {row["question_group_id"] for row in all_rows}
        ),
        "qualified_trajectory_count": sum(row["trajectory_id"] is not None for row in all_rows),
        "selected_count": len(public_samples),
        "selected_question_group_count": len(
            {row["question_group_id"] for row in public_samples}
        ),
        "selected_trajectory_count": sum(
            row["trajectory_id"] is not None for row in public_samples
        ),
        "exclusions": sorted(exclusions, key=lambda row: row["source_record_id"]),
        "strata": strata,
        "identity_reviews": identity_reviews,
        "processbench_labels": sorted(processbench_labels, key=lambda row: row["trajectory_id"]),
        "processbench_label_convention": {
            "0": "first erroneous step at zero-based index 0",
            "n": "first erroneous step at zero-based index n",
            "-1": "no erroneous step",
        },
    }
    if plan.study == "ms3":
        selected_labels = {
            row["trajectory_id"]: row["label"]
            for row in processbench_labels
            if row["meaning"] != "invalid"
        }
        audit_body["study_pools"] = {
            "diagnosis": sorted(row["sample_id"] for row in selected),
            "repair": sorted(
                row["sample_id"]
                for row in selected
                if selected_labels.get(row["trajectory_id"], -1) >= 0
            ),
        }
    audit_body["answer_key_fingerprint"] = fingerprint(answer_key)
    manifest_body: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study": plan.study,
        "phase": plan.phase,
        "rule_version": plan.rule_version,
        "identity_rule_version": IDENTITY_RULE_VERSION,
        "sources": source_metadata,
        "sampling": {
            "seed": plan.seed,
            **(
                {"quotas": dict(sorted(plan.quotas.items()))}
                if plan.study == "ms2"
                else {}
            ),
        },
        "samples": public_samples,
    }
    manifest_fingerprint = fingerprint(manifest_body)
    manifest = {
        **manifest_body,
        "manifest_id": f"{plan.study}-{plan.phase}-{manifest_fingerprint[:16]}",
        "fingerprint": manifest_fingerprint,
    }
    audit = {**audit_body, "manifest_fingerprint": manifest_fingerprint}
    return PreparedManifest(manifest=manifest, audit=audit, answer_key=answer_key)


def phase_overlap(
    left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]
) -> list[str]:
    """Return sorted original-question groups shared by two phase sample lists."""
    left_groups = {str(row["question_group_id"]) for row in left if row.get("question_group_id")}
    right_groups = {str(row["question_group_id"]) for row in right if row.get("question_group_id")}
    return sorted(left_groups & right_groups)


def assert_phase_exclusive(
    earlier: Iterable[Mapping[str, Any]],
    confirmation: Iterable[Mapping[str, Any]],
) -> None:
    """Reject any repeated original-question group across phase sample lists."""
    overlap = phase_overlap(earlier, confirmation)
    if overlap:
        raise ValueError("phase question-group overlap: " + ", ".join(overlap))


def build_question_group_exclusion_list(
    confirmation_rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Export group-based exclusions for a later sample freeze."""
    return sorted(
        {str(row["question_group_id"]) for row in confirmation_rows if row.get("question_group_id")}
    )


def _materialize_snapshot(
    snapshot: SourceSnapshot,
    plan: SamplingPlan,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for index, record in enumerate(snapshot.records, start=1):
        row = _public_row(snapshot, record, index, plan)
        rows.append(row)
        if snapshot.source == "processbench":
            label = record.get("label")
            meaning = (
                "no_error"
                if label == -1 and not isinstance(label, bool)
                else "first_error_step_index_zero_based"
                if isinstance(label, int) and not isinstance(label, bool) and label >= 0
                else "invalid"
            )
            labels.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "trajectory_source_id": row["trajectory_source_id"],
                    "label": label,
                    "meaning": meaning,
                    "final_answer_correct": record.get("final_answer_correct"),
                }
            )
    return rows, labels


def _mark_near_duplicate_candidates(rows: list[dict[str, Any]]) -> None:
    """Mark likely near duplicates for human review without merging question groups."""
    threshold = 0.97
    reviewable = {"unreviewed", "pending", "pending_review", "unresolved", "needs_review"}
    rows_by_id = {row["source_record_id"]: row for row in rows}
    inverted_index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        if row["_identity_status"] not in reviewable:
            continue
        tokens = re.findall(r"[a-z]+|\d+|[=<>+*/^\-]", row["problem"].casefold())[:256]
        width = min(3, len(tokens))
        for start in range(max(0, len(tokens) - width + 1)):
            inverted_index[tuple(tokens[start : start + width])].add(row["source_record_id"])

    candidate_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        left_id = row["source_record_id"]
        if left_id not in rows_by_id or row["_identity_status"] not in reviewable:
            continue
        tokens = re.findall(r"[a-z]+|\d+|[=<>+*/^\-]", row["problem"].casefold())[:256]
        width = min(3, len(tokens))
        shared: dict[str, int] = defaultdict(int)
        shingles = {
            tuple(tokens[start : start + width])
            for start in range(max(0, len(tokens) - width + 1))
        }
        for shingle in shingles:
            posting = inverted_index[shingle]
            if len(posting) > 50:
                continue
            for right_id in posting:
                if right_id > left_id:
                    shared[right_id] += 1
        for right_id in sorted(shared, key=lambda item: (-shared[item], item))[:256]:
            right = rows_by_id[right_id]
            if row["question_group_id"] == right["question_group_id"]:
                continue
            comparison = SequenceMatcher(
                None, row["problem"].casefold(), right["problem"].casefold(), autojunk=False
            )
            if (
                comparison.real_quick_ratio() >= threshold
                and comparison.quick_ratio() >= threshold
                and comparison.ratio() >= threshold
            ):
                candidate_ids[left_id].add(right_id)
                candidate_ids[right_id].add(left_id)

    for record_id, candidate_records in candidate_ids.items():
        row = rows_by_id[record_id]
        row["_identity_candidate_ids"] = sorted(
            set(row["_identity_candidate_ids"]) | candidate_records
        )
        if row["_identity_status"] == "unreviewed":
            row["_identity_status"] = "pending"
        if not row.get("_identity_review_reason"):
            row["_identity_review_reason"] = "near-duplicate candidate requires human review"


def _public_row(
    snapshot: SourceSnapshot,
    record: Mapping[str, Any],
    index: int,
    plan: SamplingPlan,
) -> dict[str, Any]:
    problem = normalize_problem(record.get("problem", ""))
    native_id = _native_record_id(record)
    record_id = native_id or (
        f"file:{snapshot.snapshot_name or 'inline'}:row:{index}:problem:{problem_fingerprint(problem)[:16]}"
    )
    source_record_id = f"{snapshot.source}:{snapshot.revision}:{snapshot.split}:{record_id}"
    trajectory_source_id = record.get("trajectory_id")
    if trajectory_source_id is None and snapshot.source == "processbench":
        trajectory_source_id = record.get("id") or record_id
    trajectory_id = None
    if trajectory_source_id is not None:
        trajectory_id = "trajectory-" + hashlib.sha256(
            f"{source_record_id}:{trajectory_source_id}".encode()
        ).hexdigest()[:24]
    sample_id = "sample-" + hashlib.sha256(source_record_id.encode("utf-8")).hexdigest()[:24]
    public_row = {
        "sample_id": sample_id,
        "problem": problem,
        "source": snapshot.source,
        "revision": snapshot.revision,
        "split": snapshot.split,
        "source_record_id": source_record_id,
        "problem_fingerprint": problem_fingerprint(problem),
        "question_group_id": _question_group_id(record, problem),
        "trajectory_id": trajectory_id,
        "trajectory_source_id": str(trajectory_source_id) if trajectory_source_id is not None else None,
        "match_basis": str(
            record.get(
                "match_basis",
                "upstream_question_group_mapping"
                if any(
                    record.get(key) is not None and str(record.get(key)).strip()
                    for key in (
                        "question_group_id",
                        "original_question_group_id",
                        "canonical_question_id",
                    )
                )
                else "normalized_problem_exact",
            )
        ),
        "study": plan.study,
        "phase": plan.phase,
        "sampling_stratum": (
            "processbench-blinded"
            if snapshot.source == "processbench"
            else _stratum(snapshot, record)
        ),
        "_sampling_stratum": _stratum(snapshot, record),
        "seed": plan.seed,
        "evaluation_rule_version": ANSWER_RULE_VERSION,
        "_restricted_record": dict(record),
        "_source_native_id": native_id,
        "_identity_status": str(record.get("identity_status", "unreviewed")).lower(),
        "_identity_review_reason": record.get("identity_review_reason"),
        "_identity_candidate_ids": (
            [str(value) for value in record["identity_candidate_ids"]]
            if isinstance(record.get("identity_candidate_ids"), (list, tuple))
            else [str(record["identity_candidate_ids"])]
            if record.get("identity_candidate_ids")
            else []
        ),
    }
    if snapshot.source == "processbench":
        raw_steps = record.get("steps")
        if isinstance(raw_steps, (list, tuple)):
            public_row["steps"] = list(raw_steps)
        elif isinstance(raw_steps, str):
            public_row["steps"] = [step for step in raw_steps.splitlines() if step.strip()]
    return public_row


def _without_private_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _answer_key_row(row: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "source_record_id": row["source_record_id"],
    }
    if row["source"] == "processbench":
        result.update(
            {
                "reference_material": record.get("reference_material"),
                "expert_label": record["label"],
                "steps": _preserved_processbench_steps(record.get("steps")),
                "final_answer_correct": record["final_answer_correct"],
            }
        )
    else:
        result.update(
            {"reference_answer": record["answer"], "reference_solution": record.get("solution")}
        )
    return result


def _qualification_reasons(
    source: str, record: Mapping[str, Any], row: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    identity_status = str(row.get("_identity_status", record.get("identity_status", "unreviewed"))).lower()
    if identity_status in {"pending", "pending_review", "unresolved", "needs_review"} or row.get(
        "_identity_candidate_ids"
    ):
        reasons.append("identity_pending_review")
    if not normalize_problem(record.get("problem", "")):
        reasons.append("missing_problem")
    if source in {"math500", "omni_math"} and not _has_reference_answer(record.get("answer")):
        reasons.append("missing_reference_answer")
    if source == "processbench":
        from .processbench import validate_processbench_record

        valid, process_reasons = validate_processbench_record(record)
        if not valid:
            reasons.extend(process_reasons)
    return reasons


def _has_reference_answer(value: Any) -> bool:
    return (
        isinstance(value, str) and bool(value.strip())
    ) or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _question_group_id(record: Mapping[str, Any], problem: str) -> str:
    for key in ("question_group_id", "original_question_group_id", "canonical_question_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"problem-{problem_fingerprint(problem)}"


def _stratum(snapshot: SourceSnapshot, record: Mapping[str, Any]) -> str:
    source = snapshot.source
    if source == "math500":
        image = "image" if record.get("has_image") or record.get("image") else "no_image"
        return format_stratum(
            [source, record.get("subject", "unknown"), record.get("level", "unknown"), image]
        )
    if source == "omni_math":
        return format_stratum(
            [
                source,
                record.get("domain", "unknown"),
                record.get("difficulty", "unknown"),
                record.get("source", "unknown"),
            ]
        )
    label = record.get("label")
    state = "correct" if label == -1 else "error"
    subset = record.get("source_subset", record.get("subset", record.get("source", snapshot.split)))
    return format_stratum([source, subset, record.get("generator", "unknown"), state])


def _native_record_id(record: Mapping[str, Any]) -> str | None:
    for key in ("source_record_id", "unique_id", "record_id", "id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _snapshot_fingerprint(snapshot: SourceSnapshot) -> str:
    if snapshot.snapshot_digest is not None:
        return snapshot.snapshot_digest
    records = sorted(
        snapshot.records,
        key=lambda record: _native_record_id(record) or fingerprint(dict(record)),
    )
    return fingerprint(
        {
            "source": snapshot.source,
            "revision": snapshot.revision,
            "split": snapshot.split,
            "records": records,
        }
    )


def _stable_seed(seed: int, stratum: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stratum}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _exclusion(
    row: Mapping[str, Any],
    reason: str,
    detail: str,
    reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    result = {
        "source_record_id": row["source_record_id"],
        "question_group_id": row["question_group_id"],
        "trajectory_id": row["trajectory_id"],
        "sampling_stratum": row["_sampling_stratum"],
        "reason": reason,
        "detail": detail,
    }
    if reasons is not None:
        result["reasons"] = list(reasons)
    return result


def _preserved_processbench_steps(value: Any) -> list[str]:
    """Keep source step text byte-for-byte within each non-empty parsed step."""
    if isinstance(value, (list, tuple)):
        return [step for step in value if isinstance(step, str) and step.strip()]
    if isinstance(value, str):
        return [line for line in value.splitlines() if line.strip()]
    return []
