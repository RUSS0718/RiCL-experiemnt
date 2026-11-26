"""Deterministic answer extraction, normalization, comparison, and review outcomes."""

from __future__ import annotations

import ast
import json
import math
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import sympy as sp

ANSWER_RULE_VERSION = "closed-answer-v1"


class AnswerStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"
    NO_EVALUABLE_OUTPUT = "no_evaluable_output"
    BLIND_REVIEW = "blind_review"


@dataclass(frozen=True)
class AnswerDecision:
    """Machine or adjudicated decision with the frozen rule version."""

    status: AnswerStatus
    extracted_answer: str | None
    normalized_answer: str | None
    rule_version: str = ANSWER_RULE_VERSION
    reason: str | None = None
    review_outcome: str | None = None


def extract_final_answer(output: str | None) -> str | None:
    """Extract the last boxed/marked answer or a plain numeric final line."""
    if output is None or not output.strip():
        return None
    starts = list(re.finditer(r"\\boxed\s*\{", output))
    if starts:
        start = starts[-1].end()
        depth = 1
        for index in range(start, len(output)):
            if output[index] == "{":
                depth += 1
            elif output[index] == "}":
                depth -= 1
                if depth == 0:
                    return output[start:index].strip() or None
        return None

    markers = list(
        re.finditer(r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$|^\s*####\s*(.+?)\s*$", output)
    )
    if markers:
        return (markers[-1].group(1) or markers[-1].group(2) or "").strip() or None
    last = next((line.strip() for line in reversed(output.splitlines()) if line.strip()), "")
    if _looks_like_unmarked_answer(last):
        return last
    return None


def normalize_answer(answer: str | None) -> str | None:
    """Apply the frozen whitespace and common LaTeX normalization rules."""
    if answer is None:
        return None
    value = answer.strip().strip("$").strip()
    if value.startswith(r"\(") and value.endswith(r"\)"):
        value = value[2:-2].strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\cdot", "*").replace("\\times", "*")
    value = value.replace("\\pi", "pi").replace("\\infty", "oo")
    value = _replace_simple_latex_fraction(value)
    value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    value = value.replace("^", "**")
    value = re.sub(r"\\(?:,|;|!|quad|qquad)", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.replace("{", "(").replace("}", ")")
    return value or None


def answers_equivalent(left: str, right: str) -> bool | None:
    """Return a result for safe supported algebra, or None for manual review."""
    normalized_left = normalize_answer(left)
    normalized_right = normalize_answer(right)
    if not normalized_left or not normalized_right:
        return None
    if normalized_left == normalized_right:
        return True
    try:
        difference = _safe_expression(normalized_left) - _safe_expression(normalized_right)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    return bool(sp.simplify(difference) == 0)


def classify_answer(
    output: str | None,
    reference: str | float | None,
    *,
    open_proof: bool = False,
) -> AnswerDecision:
    """Classify a closed answer or route ambiguous/open-proof cases to blinded review."""
    explicit_answers = _explicit_answer_mentions(output)
    if len(explicit_answers) > 1 and any(
        answers_equivalent(explicit_answers[0][1], item[1]) is not True
        for item in explicit_answers[1:]
    ):
        latest_answer = explicit_answers[-1][1]
        return AnswerDecision(
            AnswerStatus.BLIND_REVIEW,
            latest_answer,
            normalize_answer(latest_answer),
            reason="multiple explicit answers conflict",
        )
    if output is None or not output.strip():
        return AnswerDecision(
            AnswerStatus.NO_EVALUABLE_OUTPUT,
            None,
            None,
            reason="no model output was produced",
        )
    extracted = extract_final_answer(output)
    if extracted is None:
        return AnswerDecision(
            AnswerStatus.BLIND_REVIEW,
            None,
            None,
            reason="non-empty output has no unambiguous extractable final answer",
        )
    normalized = normalize_answer(extracted)
    if open_proof:
        return AnswerDecision(
            AnswerStatus.BLIND_REVIEW,
            extracted,
            normalized,
            reason="open proof requires the frozen process rubric and blinded review",
        )
    if reference is None or not normalized:
        return AnswerDecision(
            AnswerStatus.BLIND_REVIEW,
            extracted,
            normalized,
            reason="missing or ambiguous frozen reference",
        )
    equivalent = answers_equivalent(extracted, str(reference))
    if equivalent is None:
        return AnswerDecision(
            AnswerStatus.BLIND_REVIEW,
            extracted,
            normalized,
            reason="answer is outside the frozen equivalence rules",
        )
    status = AnswerStatus.CORRECT if equivalent else AnswerStatus.INCORRECT
    return AnswerDecision(status, extracted, normalized)


def apply_review_outcome(decision: AnswerDecision, outcome: str) -> AnswerDecision:
    """Record one independent adjudication only for a case routed to blinded review."""
    if decision.status is not AnswerStatus.BLIND_REVIEW:
        raise ValueError("only a case routed to blinded review may be adjudicated")
    try:
        status = AnswerStatus(outcome)
    except ValueError as error:
        raise ValueError("review outcome must be correct, incorrect, or unknown") from error
    if status not in {AnswerStatus.CORRECT, AnswerStatus.INCORRECT, AnswerStatus.UNKNOWN}:
        raise ValueError("review outcome must be correct, incorrect, or unknown")
    return AnswerDecision(
        status,
        decision.extracted_answer,
        decision.normalized_answer,
        decision.rule_version,
        reason=decision.reason,
        review_outcome=status.value,
    )


def write_evaluation_results(
    decisions: Iterable[tuple[str, AnswerDecision]], output_dir: Path
) -> dict[str, Path]:
    """Persist one deterministic result file per blinded outcome category."""
    root = Path(output_dir)
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(f"evaluation output directory already exists: {root}")
    categories: dict[AnswerStatus, list[dict[str, Any]]] = defaultdict(list)
    for sample_id, decision in decisions:
        if not sample_id.strip():
            raise ValueError("evaluation sample_id must not be empty")
        categories[decision.status].append(
            {
                "sample_id": sample_id,
                "status": decision.status.value,
                "extracted_answer": decision.extracted_answer,
                "normalized_answer": decision.normalized_answer,
                "rule_version": decision.rule_version,
                "reason": decision.reason,
                "review_outcome": decision.review_outcome,
            }
        )
    paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=f".{root.name}-") as temp:
        staging = Path(temp)
        for status in AnswerStatus:
            rows = sorted(categories[status], key=lambda row: row["sample_id"])
            path = staging / f"{status.value}.json"
            path.write_text(
                json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            paths[status.value] = root / path.name
        staging.rename(root)
    return paths


def _explicit_answer_mentions(output: str | None) -> list[tuple[int, str]]:
    if output is None:
        return []
    mentions: list[tuple[int, str]] = []
    for match in re.finditer(r"\\boxed\s*\{", output):
        start = match.end()
        depth = 1
        for index in range(start, len(output)):
            if output[index] == "{":
                depth += 1
            elif output[index] == "}":
                depth -= 1
                if depth == 0:
                    answer = output[start:index].strip()
                    if answer:
                        mentions.append((index + 1, answer))
                    break
    marker = re.compile(
        r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$|^\s*####\s*(.+?)\s*$"
    )
    for match in marker.finditer(output):
        answer = (match.group(1) or match.group(2)).strip()
        if answer.startswith(r"\(") and answer.endswith(r"\)"):
            answer = answer[2:-2].strip()
        if not answer or re.fullmatch(r"\\boxed\s*\{.*\}", answer):
            continue
        mentions.append((match.end(), answer))
    return sorted(mentions)


def _looks_like_unmarked_answer(value: str) -> bool:
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return True
    if value in {"pi", "\\pi", "oo", "\\infty"} or re.fullmatch(r"[a-zA-Z]", value):
        return True
    if re.fullmatch(r"[\\a-zA-Z0-9_{}()+*/^.,\- ]+", value) and re.search(
        r"[0-9^*/+\-]", value
    ):
        return _safe_normalized_expression(value)
    return False


def _safe_normalized_expression(value: str) -> bool:
    normalized = normalize_answer(value)
    if not normalized:
        return False
    try:
        _safe_expression(normalized)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return False
    return True


def _replace_simple_latex_fraction(value: str) -> str:
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while match := pattern.search(value):
        value = value[: match.start()] + f"(({match.group(1)})/({match.group(2)}))" + value[match.end() :]
    return value


def _safe_expression(value: str) -> sp.Expr:
    if len(value) > 512:
        raise ValueError("answer expression is too long")
    tree = ast.parse(value, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("answer expression is too complex")

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, float):
                if not math.isfinite(node.value):
                    raise ValueError("non-finite numeric answer")
                return sp.Rational(str(node.value))
            return sp.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return sp.pi
            if node.id == "oo":
                return sp.oo
            return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = convert(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_number and abs(float(right)) <= 100:
                return left**right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            return sp.sqrt(convert(node.args[0]))
        raise ValueError("unsupported answer expression")

    return convert(tree)
