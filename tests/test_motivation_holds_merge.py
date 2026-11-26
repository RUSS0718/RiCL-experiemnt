import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from motivation_holds_merge import merge_holds


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_merge_holds_adds_unique_question_groups_and_preserves_provenance(tmp_path: Path):
    base = tmp_path / "base.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_json(base, {"all_exclusion_keys": ["old-group"], "note": "original"})
    _write_json(
        first,
        {
            "manifest_id": "first",
            "study": "ms2",
            "phase": "retest",
            "samples": [
                {"question_group_id": "new-a"},
                {"question_group_id": "new-a"},
                {"question_group_id": "new-b"},
            ],
        },
    )
    _write_json(
        second,
        {
            "manifest_id": "second",
            "samples": [{"question_group_id": "new-b"}, {"question_group_id": "old-group"}],
        },
    )

    merged = merge_holds(base, [("first", first), ("second", second)])

    assert merged["all_exclusion_keys"] == ["new-a", "new-b", "old-group"]
    assert merged["supplemental_study_holds"]["first"]["sample_rows"] == 3
    assert merged["supplemental_study_holds"]["first"]["unique_groups"] == 2
    assert merged["supplemental_study_holds"]["first"]["new_exclusion_groups"] == 2
    assert merged["supplemental_study_holds"]["second"]["already_excluded_groups"] == 2
    assert merged["merge_provenance"]["supplemental_source_sha256"]["first"]
    assert merged["note"].startswith("original")


def test_merge_holds_rejects_samples_without_canonical_group_id(tmp_path: Path):
    base = tmp_path / "base.json"
    manifest = tmp_path / "manifest.json"
    _write_json(base, {"all_exclusion_keys": []})
    _write_json(manifest, {"samples": [{"sample_id": "missing-identity"}]})

    with pytest.raises(ValueError, match="question_group_id"):
        merge_holds(base, [("bad", manifest)])
