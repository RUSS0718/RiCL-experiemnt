from ricl_experiment.evaluation.answer import AnswerStatus, classify_answer


def test_conflicting_explicit_answer_markers_are_sent_to_blinded_review():
    decision = classify_answer("Answer: 2\nAnswer: 3", "3")

    assert decision.status is AnswerStatus.BLIND_REVIEW
    assert decision.extracted_answer == "3"
    assert "multiple" in decision.reason


def test_repeated_boxed_answer_inside_final_marker_is_not_counted_twice():
    decision = classify_answer(r"Work: \boxed{2}" + "\nFinal answer: " + r"\boxed{2}", "2")

    assert decision.status is AnswerStatus.CORRECT


def test_numeric_reference_answers_are_supported():
    decision = classify_answer("Final answer: 0.5", 0.5)

    assert decision.status is AnswerStatus.CORRECT
