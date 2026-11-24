import json

from ricl_experiment.datasets.manifest import (
    PINNED_SOURCES,
    SamplingPlan,
    SourceSnapshot,
    format_stratum,
    prepare_manifest,
)


def test_processbench_labels_and_pool_membership_stay_out_of_public_manifest():
    records = [
        {
            "id": "error-trace",
            "generator": "model-a",
            "source": "math",
            "problem": "Solve 2 + 2.",
            "steps": ["Add 2 and 2.", "Return 4."],
            "label": 0,
            "final_answer_correct": False,
            "reference_material": {
                "kind": "answer",
                "content": "4",
                "source": "verified",
                "independently_verified": True,
            },
        },
        {
            "id": "correct-trace",
            "generator": "model-a",
            "source": "math",
            "problem": "Solve 3 + 3.",
            "steps": ["Add 3 and 3.", "Return 6."],
            "label": -1,
            "final_answer_correct": True,
        },
    ]
    snapshot = SourceSnapshot(
        "processbench", PINNED_SOURCES["processbench"][1], "test", tuple(records)
    )
    plan = SamplingPlan(
        "ms3",
        "confirmatory",
        23,
        {
            format_stratum(["processbench", "math", "model-a", "error"]): 1,
            format_stratum(["processbench", "math", "model-a", "correct"]): 1,
        },
    )

    prepared = prepare_manifest(snapshot, plan)
    public_json = json.dumps(prepared.manifest, sort_keys=True)

    assert "study_pools" not in public_json
    assert '"expert_label"' not in public_json
    assert '"final_answer_correct"' not in public_json
    assert '"error"' not in public_json
    assert '"correct"' not in public_json
    assert prepared.audit["study_pools"]["repair"] == [
        sample["sample_id"]
        for sample in prepared.manifest["samples"]
        if sample["trajectory_source_id"] == "error-trace"
    ]
    assert len(prepared.audit["study_pools"]["diagnosis"]) == 2
