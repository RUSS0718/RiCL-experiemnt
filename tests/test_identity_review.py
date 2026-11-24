
from ricl_experiment.datasets.manifest import (
    PINNED_SOURCES,
    SamplingPlan,
    SourceSnapshot,
    format_stratum,
    prepare_manifest,
)


def test_near_duplicate_questions_are_pending_review_not_merged():
    snapshot = SourceSnapshot(
        source="math500",
        revision=PINNED_SOURCES["math500"][1],
        split="test",
        records=(
            {
                "unique_id": "positive",
                "problem": (
                    "At a science fair, a student arranges 12 red tiles and 18 blue tiles "
                    "into equal rows of 6 tiles. Determine the number of rows if exactly "
                    "one additional blue tile is placed."
                ),
                "answer": "2",
                "subject": "Algebra",
                "level": 4,
            },
            {
                "unique_id": "negative",
                "problem": (
                    "At a science fair, a student arranges 12 red tiles and 18 blue tiles "
                    "into equal rows of 6 tiles. Determine the number of rows if exactly "
                    "two additional blue tiles are placed."
                ),
                "answer": "-2",
                "subject": "Algebra",
                "level": 4,
            },
            {
                "unique_id": "unrelated",
                "problem": "A triangle has angles 30, 60, and 90 degrees. Find its area.",
                "answer": "1",
                "subject": "Geometry",
                "level": 4,
            },
        ),
    )
    plan = SamplingPlan(
        study="ms2",
        phase="pilot",
        seed=17,
        quotas={format_stratum(["math500", "Geometry", 4, "no_image"]): 1},
    )

    prepared = prepare_manifest(snapshot, plan)

    reviews = prepared.audit["identity_reviews"]
    assert len(reviews) == 2
    assert {row["source_record_id"].rsplit(":", 1)[-1] for row in reviews} == {
        "positive",
        "negative",
    }
    assert all(len(row["candidate_source_record_ids"]) == 1 for row in reviews)
    assert all(row["reason"].startswith("near-duplicate") for row in reviews)
    assert {
        row["source_record_id"].rsplit(":", 1)[-1]
        for row in prepared.audit["exclusions"]
        if row["reason"] == "identity_pending_review"
    } == {"positive", "negative"}
    assert [row["source_record_id"].rsplit(":", 1)[-1] for row in prepared.manifest["samples"]] == [
        "unrelated"
    ]
