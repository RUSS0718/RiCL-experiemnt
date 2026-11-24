
import pytest

from ricl_experiment.cli import main
from ricl_experiment.core.model import build_request_body
from ricl_experiment.datasets.manifest import (
    PINNED_SOURCES,
    SamplingPlan,
    SourceSnapshot,
    assert_phase_exclusive,
    build_question_group_exclusion_list,
    format_stratum,
    phase_overlap,
    prepare_manifest,
    prepare_sample_manifest,
)
from ricl_experiment.datasets.processbench import (
    build_processbench_pools,
    validate_processbench_record,
)
from ricl_experiment.evaluation.answer import (
    AnswerStatus,
    apply_review_outcome,
    classify_answer,
    normalize_answer,
    write_evaluation_results,
)

MATH_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def math_snapshot(records):
    return SourceSnapshot(
        source="math500",
        revision=MATH_REVISION,
        split="test",
        records=tuple(records),
    )


def plan(quotas, *, phase="pilot", seed=17, study="ms2"):
    return SamplingPlan(
        study=study,
        phase=phase,
        seed=seed,
        quotas=quotas,
    )


def test_fixed_source_revisions_are_pinned():
    assert PINNED_SOURCES == {
        "math500": ("HuggingFaceH4/MATH-500", MATH_REVISION),
        "omni_math": (
            "KbsdJames/Omni-MATH",
            "40ba231d8f16e29ecd40e6407e2c8640145a8f62",
        ),
        "processbench": (
            "Qwen/ProcessBench",
            "3bdcd5371ed567559a78f559c01c13a6deee7604",
        ),
    }


def test_stratified_sample_is_stable_and_keeps_source_identity():
    records = [
        {
            "unique_id": f"math-{index}",
            "problem": f"  Find x + {index} = 8.\n",
            "answer": str(8 - index),
            "subject": "Algebra",
            "level": 4,
        }
        for index in range(6)
    ]
    stratum = format_stratum(["math500", "Algebra", 4, "no_image"])
    snapshot = math_snapshot(records)

    first = prepare_manifest(snapshot, plan({stratum: 3}))
    repeated = prepare_manifest(math_snapshot(list(reversed(records))), plan({stratum: 3}))

    assert first.manifest["fingerprint"] == repeated.manifest["fingerprint"]
    assert [sample["source_record_id"] for sample in first.manifest["samples"]] == [
        sample["source_record_id"] for sample in repeated.manifest["samples"]
    ]
    assert first.manifest["sampling"]["seed"] == 17
    assert len(first.manifest["samples"]) == 3
    assert all(sample["question_group_id"] for sample in first.manifest["samples"])
    assert all(sample["problem_fingerprint"] for sample in first.manifest["samples"])
    assert all("answer" not in row for row in first.manifest["samples"])


def test_unresolved_identity_is_excluded_and_audited():
    snapshot = math_snapshot(
        [
            {
                "unique_id": "pending-1",
                "problem": "Find x if x + 1 = 2",
                "answer": "1",
                "subject": "Algebra",
                "level": 4,
                "identity_status": "pending",
                "identity_review_reason": "Near-duplicate candidate with changed condition",
                "identity_candidate_ids": ["ready-1"],
            },
            {
                "unique_id": "ready-1",
                "problem": "Find x if x + 1 = 2 with x > 0",
                "answer": "3",
                "subject": "Algebra",
                "level": 4,
            },
            {
                "unique_id": "eligible-1",
                "problem": "Find x if x + 5 = 8",
                "answer": "3",
                "subject": "Algebra",
                "level": 4,
            },
        ]
    )
    stratum = format_stratum(["math500", "Algebra", 4, "no_image"])

    prepared = prepare_manifest(snapshot, plan({stratum: 1}))

    assert [row["source_record_id"] for row in prepared.manifest["samples"]] == [f"math500:{MATH_REVISION}:test:eligible-1"]
    assert any(
        row["source_record_id"] == f"math500:{MATH_REVISION}:test:pending-1"
        and row["reason"] == "identity_pending_review"
        for row in prepared.audit["exclusions"]
    )
    assert any(
        row["source_record_id"] == f"math500:{MATH_REVISION}:test:ready-1"
        and row["reason"] == "identity_pending_review"
        for row in prepared.audit["exclusions"]
    )
    assert prepared.audit["identity_reviews"][0]["candidate_source_record_ids"] == [
        f"math500:{MATH_REVISION}:test:ready-1"
    ]


def test_mismatched_revision_and_unavailable_quota_are_rejected():
    with pytest.raises(ValueError, match="pinned revision"):
        prepare_manifest(
            SourceSnapshot("math500", "latest", "test", ()),
            plan({}),
        )

    snapshot = math_snapshot(
        [{"unique_id": "one", "problem": "2+2", "answer": "4", "subject": "Algebra", "level": 4}]
    )
    with pytest.raises(ValueError, match="quota"):
        prepare_manifest(snapshot, plan({format_stratum(["math500", "Geometry", 5, "no_image"]): 1}))


def test_phase_overlap_and_confirmation_exclusion_are_group_based():
    rows = [
        {"sample_id": "pilot-a", "question_group_id": "group-a"},
        {"sample_id": "pilot-b", "question_group_id": "group-b"},
    ]
    confirmation = [
        {"sample_id": "confirm-a", "question_group_id": "group-a"},
        {"sample_id": "confirm-c", "question_group_id": "group-c"},
    ]
    ms3 = [{"sample_id": "trace-a", "question_group_id": "group-a"}]

    assert phase_overlap(rows, confirmation) == ["group-a"]
    assert phase_overlap(confirmation, ms3) == ["group-a"]
    with pytest.raises(ValueError, match="question-group overlap"):
        assert_phase_exclusive(rows, confirmation)
    assert_phase_exclusive(
        rows,
        [{"sample_id": "confirm-c", "question_group_id": "group-c"}],
    )
    assert build_question_group_exclusion_list(confirmation) == ["group-a", "group-c"]



def processbench_record(**overrides):
    record = {
        "id": "pb-1",
        "generator": "model-a",
        "problem": "Solve 2 + 2.",
        "steps": ["Add 2 and 2.", "Therefore, the answer is 4."],
        "final_answer_correct": False,
        "label": 0,
        "reference_material": {
            "kind": "answer",
            "content": "4",
            "source": "independent reference",
            "independently_verified": True,
        },
        "source": "math",
    }
    return {**record, **overrides}


def test_error_pool_is_shared_by_diagnosis_and_repair_and_labels_are_auditable():
    error = processbench_record()
    correct = processbench_record(
        id="pb-correct",
        problem="Evaluate 3 times 3.",
        label=-1,
        final_answer_correct=True,
    )

    pools = build_processbench_pools([error, correct])

    assert [row["trajectory_id"] for row in pools["diagnosis"]] == ["pb-1", "pb-correct"]
    assert [row["trajectory_id"] for row in pools["repair"]] == ["pb-1"]
    assert all("label" not in row for row in pools["diagnosis"] + pools["repair"])
    assert pools["label_audit"] == [
        {
            "trajectory_id": "pb-1",
            "label": 0,
            "meaning": "first_error_step_index_zero_based",
            "final_answer_correct": False,
        },
        {
            "trajectory_id": "pb-correct",
            "label": -1,
            "meaning": "no_error",
            "final_answer_correct": True,
        },
    ]


def test_error_pool_rejects_invalid_labels_missing_steps_and_unscorable_answers():
    assert validate_processbench_record(processbench_record(label=2))[0] is False
    assert validate_processbench_record(processbench_record(label=2))[1] == [
        "invalid_label_out_of_range"
    ]
    assert "invalid_label_out_of_range" in validate_processbench_record(
        processbench_record(label=3)
    )[1]
    assert "insufficient_steps" in validate_processbench_record(
        processbench_record(steps=["Only one step."])
    )[1]
    assert "missing_independent_scoring_basis" in validate_processbench_record(
        processbench_record(
            reference_material={
                "kind": "answer",
                "content": "",
                "source": "",
                "independently_verified": True,
            }
        )
    )[1]
    assert validate_processbench_record(
        processbench_record(final_answer_correct=True, label=0)
    )[0]


def test_processbench_zero_and_minus_one_are_not_python_truthiness_aliases():
    error_valid, error_reasons = validate_processbench_record(processbench_record(label=0))
    correct_valid, correct_reasons = validate_processbench_record(
        processbench_record(label=-1, final_answer_correct=True)
    )

    assert error_valid and not error_reasons
    assert correct_valid and not correct_reasons
def test_frozen_answer_extraction_normalization_and_equivalence():
    equivalent = classify_answer(r"Work shown. Final answer: \boxed{\frac{1}{2}}", "0.5")
    incorrect = classify_answer("Final Answer: 7", "8")
    squared_root = classify_answer(r"\boxed{\sqrt{4}}", "2")

    assert equivalent.status is AnswerStatus.CORRECT
    assert equivalent.extracted_answer == r"\frac{1}{2}"
    assert normalize_answer(r"\left( 2 \right)") == "(2)"
    assert incorrect.status is AnswerStatus.INCORRECT
    assert squared_root.status is AnswerStatus.CORRECT


def test_ambiguous_open_proof_and_missing_output_take_distinct_review_states():
    ambiguous = classify_answer("Final answer: x = 2", "2")
    proof = classify_answer("Final answer: 2", "2", open_proof=True)
    missing = classify_answer("I cannot determine the answer.", "2")
    no_output = classify_answer(None, "2")

    assert ambiguous.status is AnswerStatus.BLIND_REVIEW
    assert proof.status is AnswerStatus.BLIND_REVIEW
    assert missing.status is AnswerStatus.BLIND_REVIEW
    assert no_output.status is AnswerStatus.NO_EVALUABLE_OUTPUT


def test_blinded_review_can_record_unknown_or_a_final_human_decision():
    pending = classify_answer("Final answer: [multiple values]", "2")

    unknown = apply_review_outcome(pending, "unknown")
    correct = apply_review_outcome(pending, "correct")

    assert unknown.status is AnswerStatus.UNKNOWN
    assert correct.status is AnswerStatus.CORRECT
    assert apply_review_outcome(pending, "incorrect").status is AnswerStatus.INCORRECT


def test_review_outcome_is_only_applied_to_a_blinded_review_case():
    decision = classify_answer("Final answer: 2", "2")

    try:
        apply_review_outcome(decision, "unknown")
    except ValueError as error:
        assert "blinded review" in str(error)
    else:
        raise AssertionError("adjudication must not overwrite an automatic decision")
def test_serialized_generation_request_contains_only_role_allowlisted_fields():
    fields = {
            "problem": "Solve 2 + 2.",
            "answer": "4",
            "solution": "The answer is 4.",
            "expert_label": 0,
            "future_outcome": "correct",
            "checkpoint": "first step",
        }

    body = build_request_body("test-model", "generation", fields, {"max_tokens": 16})

    assert body == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Solve 2 + 2."}],
        "max_tokens": 16,
    }
    serialized = repr(body)
    for restricted in ("answer", "solution", "expert_label", "future_outcome"):
        assert restricted not in serialized


def test_oracle_repair_role_receives_only_the_permitted_first_error_position():
    fields = {
            "problem": "Solve 2 + 2.",
            "first_error_position": 0,
            "expert_label": 0,
            "reference_answer": "4",
            "future_outcome": "success",
        }

    body = build_request_body("test-model", "oracle_repair", fields, {"max_tokens": 16})

    assert "First-error position:\n0" in body["messages"][0]["content"]
    serialized = repr(body)
    assert "reference_answer" not in serialized
    assert "future_outcome" not in serialized
    assert "expert_label" not in serialized
def test_prepare_cli_writes_frozen_blinded_manifest_and_separate_audit(tmp_path, capsys):
    dataset, revision = PINNED_SOURCES["math500"]
    snapshot = tmp_path / "math500.json"
    snapshot.write_text(
        __import__("json").dumps(
            {
                "source": "math500",
                "revision": revision,
                "split": "test",
                "records": [
                    {
                        "unique_id": "row-1",
                        "problem": "Solve 2+2.",
                        "answer": "4",
                        "subject": "Algebra",
                        "level": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        __import__("json").dumps(
            {
                "study": "ms2",
                "phase": "pilot",
                "seed": 11,
                "quotas": {format_stratum(["math500", "Algebra", 4, "no_image"]): 1},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "frozen"

    assert main(
        ["prepare", "--snapshot", str(snapshot), "--plan", str(plan_path), "--output", str(output)]
    ) == 0

    cli_output = capsys.readouterr().out
    manifest_path = output / "manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    audit = __import__("json").loads(
        (output / "restricted" / "audit.json").read_text(encoding="utf-8")
    )
    answer_key = __import__("json").loads(
        (output / "restricted" / "answer-key.json").read_text(encoding="utf-8")
    )
    assert f"manifest={manifest_path}" in cli_output
    assert manifest["sources"][0]["dataset"] == dataset
    assert len(manifest["sources"][0]["snapshot_fingerprint"]) == 64
    assert len(manifest["fingerprint"]) == 64
    assert "answer" not in manifest["samples"][0]
    assert answer_key[0]["reference_answer"] == "4"
    assert "answer_key=" in cli_output
    assert audit["selected_count"] == 1
    assert audit["manifest_fingerprint"] == manifest["fingerprint"]


def test_evaluation_artifacts_separate_all_required_outcome_states(tmp_path):
    decisions = [
        ("s-correct", classify_answer("Final answer: 2", "2")),
        ("s-incorrect", classify_answer("Final answer: 3", "2")),
        (
            "s-unknown",
            apply_review_outcome(classify_answer("Final answer: x=2", "2"), "unknown"),
        ),
        ("s-empty", classify_answer(None, "2")),
        ("s-review", classify_answer("Final answer: 2", "2", open_proof=True)),
    ]

    paths = write_evaluation_results(decisions, tmp_path / "eval")

    assert set(paths) == {
        "correct", "incorrect", "unknown", "no_evaluable_output", "blind_review"
    }
    for name in paths:
        assert paths[name].exists()
    assert __import__("json").loads(paths["unknown"].read_text())[0]["sample_id"] == "s-unknown"


def test_multiple_conflicting_boxed_answers_are_blinded_for_review():
    decision = classify_answer(r"Step result: \boxed{2}. Final answer: \boxed{3}", "3")

    assert decision.status is AnswerStatus.BLIND_REVIEW
    assert "multiple" in decision.reason


def test_processbench_manifest_keeps_trace_ids_and_restricts_labels_to_sidecar():
    revision = PINNED_SOURCES["processbench"][1]
    error = processbench_record(
        question_group_id="shared-question",
        reference_material={
            "kind": "answer",
            "content": "4",
            "source": "reviewed-source",
            "independently_verified": True,
        },
    )
    correct_trace = processbench_record(
        id="pb-correct",
        problem="Evaluate 3 times 3.",
        question_group_id="another-question",
        label=-1,
        final_answer_correct=True,
    )
    second_trace = processbench_record(
        id="pb-2",
        question_group_id="shared-question",
        steps=["Add 2 and 2.", "Check the sum.", "Return 4."],
        label=1,
        reference_material={
            "kind": "answer",
            "content": "4",
            "source": "reviewed-source",
            "independently_verified": True,
        },
    )
    stratum = format_stratum(["processbench", "math", "model-a", "error"])
    prepared = prepare_sample_manifest(
        [SourceSnapshot("processbench", revision, "test", (error, second_trace, correct_trace))],
        plan(
            {
                stratum: 2,
                format_stratum(["processbench", "math", "model-a", "correct"]): 1,
            },
            study="ms3",
            phase="confirmatory",
        ),
    )

    samples = prepared.manifest["samples"]
    error_samples = [sample for sample in samples if sample["question_group_id"] == "shared-question"]
    assert len({sample["trajectory_id"] for sample in error_samples}) == 2
    assert len(samples) == 3
    assert {sample["question_group_id"] for sample in error_samples} == {"shared-question"}
    assert all("expert_label" not in sample and "reference_material" not in sample for sample in samples)
    assert {row["expert_label"] for row in prepared.answer_key} == {-1, 0, 1}
    assert len(prepared.audit["study_pools"]["diagnosis"]) == 3
    assert len(prepared.audit["study_pools"]["repair"]) == 2
    assert set(prepared.audit["study_pools"]["repair"]).issubset(
        prepared.audit["study_pools"]["diagnosis"]
    )
    assert {row["meaning"] for row in prepared.audit["processbench_labels"]} == {
        "first_error_step_index_zero_based", "no_error"
    }
    assert prepared.audit["processbench_label_convention"]["0"].endswith("index 0")
    assert prepared.audit["processbench_label_convention"]["-1"] == "no erroneous step"


def test_phase_exclusion_prevents_selecting_any_trace_from_observed_question():
    snapshot = SourceSnapshot(
        "math500",
        PINNED_SOURCES["math500"][1],
        "test",
        (
            {
                "unique_id": "held-out",
                "problem": "Solve 1 + 1.",
                "answer": "2",
                "subject": "Algebra",
                "level": 4,
                "question_group_id": "q-1",
            },
            {
                "unique_id": "eligible",
                "problem": "Solve 3 + 4.",
                "answer": "7",
                "subject": "Algebra",
                "level": 4,
                "question_group_id": "q-2",
            },
        ),
    )
    stratum = format_stratum(["math500", "Algebra", 4, "no_image"])
    confirm = prepare_manifest(
        snapshot,
        SamplingPlan(
            "ms2", "confirmatory", 3, {stratum: 1}, excluded_question_group_ids=("q-1",)
        ),
    )
    assert_phase_exclusive(
        [{"question_group_id": "q-1"}],
        confirm.manifest["samples"],
    )
    excluded = prepare_manifest(
        snapshot,
        SamplingPlan(
            "ms2",
            "pilot",
            5,
            {stratum: 1},
            excluded_question_group_ids=("q-1",),
        ),
    )
    assert [sample["question_group_id"] for sample in excluded.manifest["samples"]] == ["q-2"]
    assert any(row["reason"] == "question_group_excluded" for row in excluded.audit["exclusions"])


def test_overlap_export_is_same_phase_ms2_ms3_and_preserves_group_ids(tmp_path):
    from ricl_experiment.datasets.manifest import write_overlap_report

    left_path = tmp_path / "ms2.json"
    right_path = tmp_path / "ms3.json"
    output_path = tmp_path / "overlap.json"
    left_path.write_text(
        __import__("json").dumps(
            {
                "study": "ms2",
                "phase": "pilot",
                "fingerprint": "left-fingerprint",
                "samples": [
                    {"question_group_id": "shared"},
                    {"question_group_id": "only-ms2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    right_path.write_text(
        __import__("json").dumps(
            {
                "study": "ms3",
                "phase": "pilot",
                "fingerprint": "right-fingerprint",
                "samples": [{"question_group_id": "shared"}],
            }
        ),
        encoding="utf-8",
    )

    write_overlap_report(left_path, right_path, output_path)

    report = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    assert report["overlap_count"] == 1
    assert report["question_group_ids"] == ["shared"]


def test_overlap_export_rejects_different_stages_and_replacement(tmp_path):
    from ricl_experiment.datasets.manifest import write_overlap_report

    left_path = tmp_path / "ms2.json"
    right_path = tmp_path / "ms3.json"
    for path, study, phase in (
        (left_path, "ms2", "pilot"),
        (right_path, "ms3", "confirmatory"),
    ):
        path.write_text(
            __import__("json").dumps({"study": study, "phase": phase, "samples": []}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="same phase"):
        write_overlap_report(left_path, right_path, tmp_path / "report.json")

    right_path.write_text(
        __import__("json").dumps({"study": "ms3", "phase": "pilot", "samples": []}),
        encoding="utf-8",
    )
    destination = tmp_path / "existing.json"
    destination.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_overlap_report(left_path, right_path, destination)
    assert destination.read_text(encoding="utf-8") == "keep"


def test_exclusion_export_requires_ms2_confirmation_and_is_unique(tmp_path):
    from ricl_experiment.datasets.manifest import export_question_group_exclusions

    manifest = tmp_path / "confirm.json"
    manifest.write_text(
        __import__("json").dumps(
            {
                "study": "ms2",
                "phase": "confirmatory",
                "samples": [
                    {"question_group_id": "q1"},
                    {"question_group_id": "q1"},
                    {"question_group_id": "q2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "exclusions.json"
    export_question_group_exclusions(manifest, output)
    assert __import__("json").loads(output.read_text(encoding="utf-8")) == ["q1", "q2"]

    with pytest.raises(FileExistsError):
        export_question_group_exclusions(manifest, output)


def test_processbench_falls_back_to_id_when_trajectory_id_is_null():
    record = processbench_record(trajectory_id=None, id="fallback-id")

    valid, reasons = validate_processbench_record(record)
    pools = build_processbench_pools([record])

    assert valid and not reasons
    assert pools["diagnosis"][0]["trajectory_id"] == "fallback-id"


def test_unresolved_identity_candidates_remain_in_review_audit(tmp_path):
    snapshot_path = tmp_path / "math500.json"
    snapshot_path.write_text(
        __import__("json").dumps(
            {
                "source": "math500",
                "revision": MATH_REVISION,
                "split": "test",
                "records": [
                    {
                        "unique_id": "pending",
                        "problem": "Find x when x + 1 = 2.",
                        "answer": "1",
                        "subject": "Algebra",
                        "level": 4,
                        "identity_status": "pending",
                        "identity_review_reason": "External near-duplicate needs review.",
                        "identity_candidate_ids": ["omni_math:record-42"],
                    },
                    {
                        "unique_id": "eligible",
                        "problem": "Find x when x + 2 = 5.",
                        "answer": "3",
                        "subject": "Algebra",
                        "level": 4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot = __import__("ricl_experiment.datasets.manifest", fromlist=["load_snapshot"]).load_snapshot(snapshot_path)
    prepared = prepare_manifest(
        snapshot,
        plan({format_stratum(["math500", "Algebra", 4, "no_image"]): 1}),
    )

    assert prepared.audit["identity_reviews"][0]["unresolved_candidate_ids"] == [
        "omni_math:record-42"
    ]


def test_processbench_text_steps_ignore_blank_line_separators_but_arrays_do_not():
    text_trace = processbench_record(steps="First step.\n\nSecond step.")
    malformed_trace = processbench_record(steps=["First step.", "  ", "Second step."])

    assert validate_processbench_record(text_trace)[0]
    assert "insufficient_steps" in validate_processbench_record(malformed_trace)[1]
