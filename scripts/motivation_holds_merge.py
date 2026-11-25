from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


def _read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload, raw


def merge_holds(base_path: Path, manifests: Iterable[tuple[str, Path]]) -> dict:
    """Add every frozen cohort's question groups to the Main Experiment hold list."""
    base, base_raw = _read_json(base_path)
    existing = base.get("all_exclusion_keys")
    if not isinstance(existing, list) or any(not isinstance(group, str) for group in existing):
        raise ValueError("base all_exclusion_keys must be a list of strings")

    output = dict(base)
    supplements = dict(output.get("supplemental_study_holds", {}))
    all_groups = set(existing)
    provenance = {}
    for name, path in manifests:
        if name in supplements or name in provenance:
            raise ValueError(f"duplicate hold source name: {name}")
        manifest, raw = _read_json(path)
        samples = manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"manifest has no samples: {path}")

        group_ids = []
        for sample in samples:
            group_id = sample.get("question_group_id") if isinstance(sample, dict) else None
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"every sample must have question_group_id: {path}")
            group_ids.append(group_id)
        groups = set(group_ids)
        prior = all_groups.copy()
        all_groups.update(groups)
        supplements[name] = {
            "source_path": path.as_posix(),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_id": manifest.get("manifest_id"),
            "study": manifest.get("study"),
            "phase": manifest.get("phase"),
            "sample_rows": len(samples),
            "unique_groups": len(groups),
            "already_excluded_groups": len(groups & prior),
            "new_exclusion_groups": len(groups - prior),
            "group_ids": sorted(groups),
        }
        provenance[name] = hashlib.sha256(raw).hexdigest()

    output["supplemental_study_holds"] = supplements
    output["all_exclusion_keys"] = sorted(all_groups)
    output["merge_provenance"] = {
        "base_source_path": base_path.as_posix(),
        "base_source_sha256": hashlib.sha256(base_raw).hexdigest(),
        "supplemental_source_sha256": provenance,
    }
    note = output.get("note", "")
    output["note"] = (
        f"{note} Supplemental frozen continuation cohorts are held out regardless "
        "of whether their planned API requests were dispatched."
    ).strip()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True, metavar="NAME=PATH")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    manifests = []
    for item in args.manifest:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --manifest {item!r}; expected NAME=PATH")
        manifests.append((name, Path(raw_path)))

    result = merge_holds(args.base, manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "total_exclusion_keys": len(result["all_exclusion_keys"]),
                "supplemental_sources": len(result["supplemental_study_holds"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
