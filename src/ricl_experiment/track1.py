"""Track 1 (Exploration / Dev) runner, dataset loader, and offline evaluation helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core.model import normalize_openai_base_url, usage_drift_warnings
from .datasets.manifest import normalize_problem, problem_fingerprint
from .evaluation.answer import (
    AnswerStatus,
    answers_equivalent,
    classify_answer,
    extract_final_answer,
    normalize_answer,
    _safe_expression,
)

# Align local USD rate with PR #273 Track1Provider (cn2.ezaigc.cn advertised group rate).
INPUT_USD_PER_MILLION = Decimal("0.80")
OUTPUT_USD_PER_MILLION = Decimal("3.20")

SOLVE_PROMPT_SUFFIX = (
    "\n\nSolve the problem step by step carefully, and put ONLY the final answer"
    " value or expression (without an equation or left-hand side) inside \\boxed{...}."
)


@dataclass(frozen=True)
class CandidateProblem:
    """Normalized competition problem with separated reference answer."""

    problem_id: str
    source: str
    difficulty_tag: str
    question_group_id: str
    problem: str
    reference_answer: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "difficulty_tag": self.difficulty_tag,
            "question_group_id": self.question_group_id,
            "problem": self.problem,
        }

    def answer_key_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "question_group_id": self.question_group_id,
            "reference_answer": self.reference_answer,
        }


def is_sympy_auto_evaluable(answer: Any) -> bool:
    """Return True if answer is a mathematical expression parseable by answer.py."""
    if answer is None or isinstance(answer, bool):
        return False
    text = str(answer).strip()
    if not text:
        return False
    # Reject bare English words like "Yes", "No", "Converges" that ast.Name treats as symbols.
    if re.fullmatch(r"[A-Za-z]{2,}", text) and text not in {"pi", "oo"}:
        return False
    norm = normalize_answer(text)
    if not norm:
        return False
    try:
        _safe_expression(norm)
        return True
    except Exception:
        return False


def _stable_hash(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def load_competition_candidates(
    repo_root: Path,
    *,
    math500_level: int = 5,
    omnimath_min_difficulty: float = 5.0,
) -> dict[str, list[CandidateProblem]]:
    """Load SymPy-auto-evaluable candidates from local MATH-500, Omni-MATH, and OlympiadBench snapshots."""
    math_path = repo_root / "data" / "snapshots" / "math500-test.json"
    omni_path = repo_root / "data" / "snapshots" / "omni_math-test.json"
    oly_path = repo_root / "data" / "ms3-reference-review" / "processbench-olympiadbench.json"

    math_records = json.loads(math_path.read_text(encoding="utf-8"))["records"]
    omni_records = json.loads(omni_path.read_text(encoding="utf-8"))["records"]
    oly_records = json.loads(oly_path.read_text(encoding="utf-8"))["records"]

    math_pool: list[CandidateProblem] = []
    for row in math_records:
        if row.get("level") != math500_level:
            continue
        ans = str(row.get("answer", "")).strip()
        if not is_sympy_auto_evaluable(ans):
            continue
        prob = normalize_problem(row.get("problem"))
        if not prob or "TNYWR" in prob or "T N Y W R" in prob:
            continue
        uid = str(row.get("unique_id") or problem_fingerprint(prob)[:16])
        pid = f"math500__{re.sub(r'[^A-Za-z0-9_-]+', '_', uid).strip('_')}"
        math_pool.append(
            CandidateProblem(
                problem_id=pid,
                source="math500",
                difficulty_tag=f"level_{row.get('level')}_{row.get('subject', 'unknown')}",
                question_group_id=str(row.get("question_group_id") or f"math500:{uid}"),
                problem=prob,
                reference_answer=ans,
            )
        )

    omni_pool: list[CandidateProblem] = []
    for row in omni_records:
        diff = row.get("difficulty")
        if not isinstance(diff, (int, float)) or diff < omnimath_min_difficulty:
            continue
        ans = str(row.get("answer", "")).strip()
        if not is_sympy_auto_evaluable(ans):
            continue
        prob = normalize_problem(row.get("problem"))
        if not prob or "TNYWR" in prob or "T N Y W R" in prob:
            continue
        uid = str(row.get("unique_id") or problem_fingerprint(prob)[:16])
        pid = f"omnimath__{re.sub(r'[^A-Za-z0-9_-]+', '_', uid).strip('_')}"
        omni_pool.append(
            CandidateProblem(
                problem_id=pid,
                source="omni_math",
                difficulty_tag=f"diff_{diff}",
                question_group_id=str(row.get("question_group_id") or f"omni_math:{uid}"),
                problem=prob,
                reference_answer=ans,
            )
        )

    oly_pool: list[CandidateProblem] = []
    seen_oly_groups: set[str] = set()
    for row in oly_records:
        if row.get("final_answer_correct") is not True:
            continue
        steps = row.get("steps") or []
        last_step = steps[-1] if isinstance(steps, list) and steps else ""
        ans = extract_final_answer(last_step)
        if not is_sympy_auto_evaluable(ans):
            continue
        prob = normalize_problem(row.get("problem"))
        if not prob or "TNYWR" in prob or "T N Y W R" in prob:
            continue
        qgid = f"problem-{problem_fingerprint(prob)}"
        if qgid in seen_oly_groups:
            continue
        seen_oly_groups.add(qgid)
        uid = str(row.get("id") or problem_fingerprint(prob)[:16])
        pid = f"olympiadbench__{re.sub(r'[^A-Za-z0-9_-]+', '_', uid).strip('_')}"
        oly_pool.append(
            CandidateProblem(
                problem_id=pid,
                source="olympiadbench",
                difficulty_tag="processbench_olympiadbench",
                question_group_id=qgid,
                problem=prob,
                reference_answer=str(ans),
            )
        )

    return {
        "math500": math_pool,
        "omni_math": omni_pool,
        "olympiadbench": oly_pool,
    }


def estimate_usd(prompt_tokens: int | None, completion_tokens: int | None) -> Decimal:
    """Compute local estimated USD from input/output tokens."""
    in_tok = Decimal(prompt_tokens or 0)
    out_tok = Decimal(completion_tokens or 0)
    return (
        in_tok * INPUT_USD_PER_MILLION + out_tok * OUTPUT_USD_PER_MILLION
    ) / Decimal(1_000_000)


def extract_usage(payload: Mapping[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Extract prompt_tokens, completion_tokens, and reasoning_tokens from an API payload."""
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, None
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    prompt_tokens = pt if isinstance(pt, int) and not isinstance(pt, bool) and pt >= 0 else None
    completion_tokens = ct if isinstance(ct, int) and not isinstance(ct, bool) and ct >= 0 else None
    details = usage.get("completion_tokens_details")
    rt = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    reasoning_tokens = rt if isinstance(rt, int) and not isinstance(rt, bool) and rt >= 0 else None
    return prompt_tokens, completion_tokens, reasoning_tokens


def call_chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int = 16384,
    temperature: float | None = None,
    thinking_type: str = "disabled",
    reasoning_effort: str | None = "none",
    json_mode: bool = False,
    timeout_seconds: float = 180.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Call OpenAI-compatible chat/completions matching PR #273 Track 1 config."""
    if not json_mode and max_tokens < 8192:
        raise ValueError(f"Track 1 solve calls require max_tokens >= 8192, got {max_tokens}")
    normalized_base = normalize_openai_base_url(base_url)
    endpoint = f"{normalized_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "thinking": {"type": thinking_type},
        "stream": False,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    attempts_used = 0
    last_error = ""
    raw_payload: dict[str, Any] | None = None
    elapsed_s = 0.0

    for attempt_idx in range(1, max_attempts + 1):
        attempts_used = attempt_idx
        started = time.monotonic()
        try:
            with httpx.Client(timeout=timeout_seconds, trust_env=False, proxy=proxy) as client:
                response = client.post(endpoint, json=body, headers=headers)
            elapsed_s = time.monotonic() - started
            if response.is_success:
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get("choices"), list) and payload["choices"]:
                    raw_payload = payload
                    break
                last_error = "invalid_completion_shape"
            else:
                last_error = f"HTTP_{response.status_code}"
                if response.status_code not in {429} and response.status_code < 500:
                    break
        except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
            elapsed_s = time.monotonic() - started
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt_idx < max_attempts:
            time.sleep(min(2.0 * attempt_idx, 5.0))

    if raw_payload is None:
        return {
            "ok": False,
            "error": last_error or "unknown_error",
            "attempts_used": attempts_used,
            "wall_time_s": round(elapsed_s, 3),
            "content": "",
            "reasoning_content": "",
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
            "warnings": [],
            "estimated_usd": "0",
        }

    choice = raw_payload["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    reasoning_content = (
        message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else ""
    )
    finish_reason = choice.get("finish_reason")
    prompt_tokens, completion_tokens, reasoning_tokens = extract_usage(raw_payload)
    warnings = usage_drift_warnings(completion_tokens, reasoning_tokens, max_tokens=max_tokens)
    total_tokens = (
        (prompt_tokens or 0) + (completion_tokens or 0)
        if prompt_tokens is not None or completion_tokens is not None
        else None
    )
    cost = estimate_usd(prompt_tokens, completion_tokens)
    return {
        "ok": True,
        "error": None,
        "attempts_used": attempts_used,
        "wall_time_s": round(elapsed_s, 3),
        "content": content,
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "warnings": warnings,
        "estimated_usd": str(cost),
    }


def _strip_outer_boxed(expr: str) -> str:
    """Unwrap outer math delimiters, trailing periods, and full \\boxed{...} layers."""
    value = expr.strip()
    for _ in range(3):
        prev = value
        value = value.strip(" *$\t\r\n").rstrip(".").strip(" *$\t\r\n")
        if value.startswith("\\(") and value.endswith("\\)"):
            value = value[2:-2].strip()
        if value.startswith("\\[") and value.endswith("\\]"):
            value = value[2:-2].strip()
        value = value.strip(" *$\t\r\n").rstrip(".").strip(" *$\t\r\n")
        if value.startswith("\\boxed{"):
            start = len("\\boxed{")
            depth = 1
            pos = start
            while pos < len(value) and depth > 0:
                if value[pos] == "{":
                    depth += 1
                elif value[pos] == "}":
                    depth -= 1
                pos += 1
            if depth == 0 and pos == len(value):
                value = value[start : pos - 1].strip()
        if value == prev:
            break
    return value


def format_boxed_answer(ans: str) -> str:
    """Wrap an answer inside \\boxed{...} without double-wrapping an outer box."""
    inner = _strip_outer_boxed(ans)
    return f"\\boxed{{{inner or ans.strip()}}}"


def _pre_normalize_latex_expr(expr: str) -> str:
    """Normalize common shorthand LaTeX before calling frozen answer.py rules."""
    value = _strip_outer_boxed(expr)
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = re.sub(r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])", r"\\frac{\1}{\2}", value)
    value = re.sub(r"\\sqrt\s*([0-9A-Za-z])", r"\\sqrt{\1}", value)
    # Replace nested \sqrt{...} before fraction replacement so answer.py can parse \frac{a}{\sqrt{b}}.
    value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    return value


def evaluate_output(content: str, reference_answer: str) -> dict[str, Any]:
    """Run offline answer evaluation using src/ricl_experiment/evaluation/answer.py."""
    decision = classify_answer(content, reference_answer)
    if decision.status is AnswerStatus.BLIND_REVIEW and decision.extracted_answer:
        pre_ext = _pre_normalize_latex_expr(decision.extracted_answer)
        pre_ref = _pre_normalize_latex_expr(reference_answer)
        fallback = classify_answer(format_boxed_answer(pre_ext), pre_ref)
        if fallback.status in {AnswerStatus.CORRECT, AnswerStatus.INCORRECT}:
            return {
                "status": fallback.status.value,
                "extracted_answer": decision.extracted_answer,
                "normalized_answer": fallback.normalized_answer,
                "reason": fallback.reason,
            }
    return {
        "status": decision.status.value,
        "extracted_answer": decision.extracted_answer,
        "normalized_answer": decision.normalized_answer,
        "reason": decision.reason,
    }


def run_smoke(
    repo_root: Path,
    output_dir: Path,
    *,
    max_tokens: int = 16384,
    temperature: float | None = None,
    thinking_type: str = "disabled",
    reasoning_effort: str | None = "none",
    max_requests: int = 6,
    max_estimated_usd: Decimal = Decimal("0.20"),
) -> dict[str, Any]:
    """Step 1: Run 3-problem API smoke across MATH L5, Omni-MATH, and OlympiadBench."""
    base_url = os.environ.get("BASE_URL", "")
    model = os.environ.get("MODEL", "")
    api_key = os.environ.get("API_KEY", "")
    if not base_url or not model or not api_key:
        raise RuntimeError("BASE_URL, MODEL, and API_KEY environment variables must be set")

    pools = load_competition_candidates(repo_root)
    selected = [
        pools["math500"][0],
        pools["omni_math"][0],
        pools["olympiadbench"][0],
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    requests_sent = 0
    total_usd = Decimal("0")

    for prob in selected:
        if requests_sent >= max_requests:
            raise RuntimeError(f"max_requests cap ({max_requests}) reached before completing smoke")
        if total_usd >= max_estimated_usd:
            raise RuntimeError(f"max_estimated_usd cap ({max_estimated_usd}) reached")

        remaining_attempts = max(1, min(2, max_requests - requests_sent))
        prompt = prob.problem + SOLVE_PROMPT_SUFFIX
        res = call_chat_completion(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            max_attempts=remaining_attempts,
        )
        requests_sent += res["attempts_used"]
        total_usd += Decimal(res["estimated_usd"])

        ev = evaluate_output(res["content"], prob.reference_answer) if res["ok"] else {
            "status": AnswerStatus.NO_EVALUABLE_OUTPUT.value,
            "extracted_answer": None,
            "normalized_answer": None,
            "reason": res["error"],
        }

        resp_file = responses_dir / f"{prob.problem_id}.json"
        resp_file.write_text(
            json.dumps(
                {
                    "problem_id": prob.problem_id,
                    "source": prob.source,
                    "content": res["content"],
                    "reasoning_content": res["reasoning_content"],
                    "finish_reason": res["finish_reason"],
                    "warnings": res["warnings"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        row = {
            "problem_id": prob.problem_id,
            "source": prob.source,
            "difficulty_tag": prob.difficulty_tag,
            "ok": res["ok"],
            "error": res["error"],
            "attempts_used": res["attempts_used"],
            "wall_time_s": res["wall_time_s"],
            "finish_reason": res["finish_reason"],
            "empty_content": not bool(res["content"].strip()),
            "truncated": res["finish_reason"] != "stop",
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "reasoning_tokens": res["reasoning_tokens"],
            "total_tokens": res["total_tokens"],
            "warnings": res["warnings"],
            "estimated_usd": res["estimated_usd"],
            "eval_status": ev["status"],
            "extracted_answer": ev["extracted_answer"],
            "reference_answer": prob.reference_answer,
            "response_ref": str(resp_file.relative_to(output_dir)),
        }
        rows.append(row)

    with records_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    total_completion = sum(r["completion_tokens"] or 0 for r in rows)
    total_reasoning = sum(r["reasoning_tokens"] or 0 for r in rows)
    empty_count = sum(1 for r in rows if r["empty_content"])
    truncated_count = sum(1 for r in rows if r["truncated"])
    warning_count = sum(len(r["warnings"]) for r in rows)

    summary = {
        "stage": "smoke",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": normalize_openai_base_url(base_url),
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "problems_tested": len(rows),
        "requests_sent": requests_sent,
        "ok_count": sum(1 for r in rows if r["ok"]),
        "empty_content_count": empty_count,
        "empty_content_rate": empty_count / len(rows),
        "unexpected_truncation_count": truncated_count,
        "unexpected_truncation_rate": truncated_count / len(rows),
        "warning_count": warning_count,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "reasoning_tokens": total_reasoning,
        "total_tokens": total_prompt + total_completion,
        "estimated_usd": str(total_usd),
        "eval_counts": dict(Counter(r["eval_status"] for r in rows)),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def sample_screening_candidates(
    repo_root: Path,
    *,
    per_source: int = 24,
    omnimath_min_difficulty: float = 7.0,
) -> list[CandidateProblem]:
    """Deterministically sample a balanced hard candidate pool across the 3 local sources."""
    pools = load_competition_candidates(
        repo_root,
        math500_level=5,
        omnimath_min_difficulty=omnimath_min_difficulty,
    )
    selected: list[CandidateProblem] = []
    seen_groups: set[str] = set()
    seen_fps: set[str] = set()
    for source in ("math500", "omni_math", "olympiadbench"):
        ordered = sorted(
            pools[source],
            key=lambda item: _stable_hash("track1-exp35-screen-v1", item.problem_id),
        )
        picked = 0
        for item in ordered:
            fp = problem_fingerprint(item.problem)
            if item.question_group_id in seen_groups or fp in seen_fps:
                continue
            seen_groups.add(item.question_group_id)
            seen_fps.add(fp)
            selected.append(item)
            picked += 1
            if picked >= per_source:
                break
        if picked < per_source:
            raise RuntimeError(
                f"Requested {per_source} candidates from {source}, only found {picked}"
            )
    return selected


def write_problem_split(
    problems: Sequence[CandidateProblem],
    output_dir: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist public problem statements separately from offline evaluation answer keys."""
    output_dir.mkdir(parents=True, exist_ok=True)
    problems_path = output_dir / "problems.jsonl"
    answer_key_path = output_dir / "answer_key.jsonl"
    with problems_path.open("w", encoding="utf-8") as pf:
        for item in problems:
            pf.write(json.dumps(item.public_dict(), ensure_ascii=False) + "\n")
    with answer_key_path.open("w", encoding="utf-8") as af:
        for item in problems:
            af.write(json.dumps(item.answer_key_dict(), ensure_ascii=False) + "\n")
    if metadata is not None:
        (output_dir / "metadata.json").write_text(
            json.dumps(dict(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return {"problems": problems_path, "answer_key": answer_key_path}


def load_problem_split(split_dir: Path) -> list[CandidateProblem]:
    """Load public problems and answer keys back into CandidateProblem objects for grading."""
    problems_rows = [
        json.loads(line)
        for line in (split_dir / "problems.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    key_rows = {
        row["problem_id"]: row
        for line in (split_dir / "answer_key.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    merged: list[CandidateProblem] = []
    for row in problems_rows:
        key = key_rows[row["problem_id"]]
        merged.append(
            CandidateProblem(
                problem_id=row["problem_id"],
                source=row["source"],
                difficulty_tag=row["difficulty_tag"],
                question_group_id=row["question_group_id"],
                problem=row["problem"],
                reference_answer=key["reference_answer"],
            )
        )
    return merged


def run_screen(
    repo_root: Path,
    output_dir: Path,
    devset_dir: Path,
    *,
    per_source: int = 24,
    omnimath_min_difficulty: float = 7.0,
    target_count: int = 30,
    max_tokens: int = 16384,
    temperature: float = 0.6,
    max_requests: int = 90,
    max_estimated_usd: Decimal = Decimal("1.00"),
    max_workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Step 2: Screen ~72 candidates with pass@1 and build the 30-problem golden devset."""
    base_url = os.environ.get("BASE_URL", "")
    model = os.environ.get("MODEL", "")
    api_key = os.environ.get("API_KEY", "")
    if not base_url or not model or not api_key:
        raise RuntimeError("BASE_URL, MODEL, and API_KEY environment variables must be set")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    if resume and (candidates_dir / "problems.jsonl").is_file():
        candidates = load_problem_split(candidates_dir)
    else:
        candidates = sample_screening_candidates(
            repo_root,
            per_source=per_source,
            omnimath_min_difficulty=omnimath_min_difficulty,
        )
        write_problem_split(
            candidates,
            candidates_dir,
            metadata={
                "stage": "screen_candidates",
                "per_source": per_source,
                "omnimath_min_difficulty": omnimath_min_difficulty,
                "total_candidates": len(candidates),
            },
        )

    existing_rows: dict[str, dict[str, Any]] = {}
    if resume and records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                existing_rows[row["problem_id"]] = row

    pending = [c for c in candidates if c.problem_id not in existing_rows]
    lock = threading.Lock()
    requests_sent = sum(r.get("attempts_used", 1) for r in existing_rows.values())
    total_usd = sum((Decimal(r.get("estimated_usd", "0")) for r in existing_rows.values()), Decimal("0"))
    rows_by_id: dict[str, dict[str, Any]] = dict(existing_rows)

    def _run_one(prob: CandidateProblem) -> dict[str, Any] | None:
        nonlocal requests_sent, total_usd
        with lock:
            if requests_sent >= max_requests or total_usd >= max_estimated_usd:
                return None
            remaining_attempts = max(1, min(2, max_requests - requests_sent))
            # Reserve 1 request slot optimistically; adjust after call returns actual attempts.
            requests_sent += 1

        res = call_chat_completion(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prob.problem + SOLVE_PROMPT_SUFFIX,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            max_attempts=remaining_attempts,
        )
        extra_attempts = max(0, res["attempts_used"] - 1)
        ev = (
            evaluate_output(res["content"], prob.reference_answer)
            if res["ok"]
            else {
                "status": AnswerStatus.NO_EVALUABLE_OUTPUT.value,
                "extracted_answer": None,
                "normalized_answer": None,
                "reason": res["error"],
            }
        )
        resp_file = responses_dir / f"{prob.problem_id}.json"
        resp_file.write_text(
            json.dumps(
                {
                    "problem_id": prob.problem_id,
                    "source": prob.source,
                    "content": res["content"],
                    "reasoning_content": res["reasoning_content"],
                    "finish_reason": res["finish_reason"],
                    "warnings": res["warnings"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        row = {
            "problem_id": prob.problem_id,
            "source": prob.source,
            "difficulty_tag": prob.difficulty_tag,
            "ok": res["ok"],
            "error": res["error"],
            "attempts_used": res["attempts_used"],
            "wall_time_s": res["wall_time_s"],
            "finish_reason": res["finish_reason"],
            "empty_content": not bool(res["content"].strip()),
            "truncated": res["finish_reason"] != "stop",
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "reasoning_tokens": res["reasoning_tokens"],
            "total_tokens": res["total_tokens"],
            "warnings": res["warnings"],
            "estimated_usd": res["estimated_usd"],
            "eval_status": ev["status"],
            "extracted_answer": ev["extracted_answer"],
            "reference_answer": prob.reference_answer,
            "eval_reason": ev["reason"],
            "response_ref": str(resp_file.relative_to(output_dir)),
        }
        with lock:
            requests_sent += extra_attempts
            total_usd += Decimal(res["estimated_usd"])
            rows_by_id[prob.problem_id] = row
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    if not resume and records_path.exists():
        records_path.unlink()

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = [pool.submit(_run_one, prob) for prob in pending]
            for fut in as_completed(futures):
                fut.result()

    ordered_rows = [rows_by_id[c.problem_id] for c in candidates if c.problem_id in rows_by_id]
    with records_path.open("w", encoding="utf-8") as fh:
        for row in ordered_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_problems, selection_info = select_golden_devset(
        ordered_rows, candidates, target_count=target_count
    )
    if selected_problems:
        write_problem_split(
            selected_problems,
            devset_dir,
            metadata={
                "stage": "track1_dev30",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **selection_info,
            },
        )

    by_source: dict[str, dict[str, Any]] = {}
    for src in ("math500", "omni_math", "olympiadbench"):
        sub = [r for r in ordered_rows if r["source"] == src]
        correct = sum(1 for r in sub if r["eval_status"] == "correct")
        by_source[src] = {
            "count": len(sub),
            "correct": correct,
            "incorrect": sum(1 for r in sub if r["eval_status"] == "incorrect"),
            "blind_review": sum(1 for r in sub if r["eval_status"] == "blind_review"),
            "no_evaluable_output": sum(1 for r in sub if r["eval_status"] == "no_evaluable_output"),
            "pass_at_1": (correct / len(sub)) if sub else 0.0,
        }

    total_prompt = sum(r["prompt_tokens"] or 0 for r in ordered_rows)
    total_completion = sum(r["completion_tokens"] or 0 for r in ordered_rows)
    total_reasoning = sum(r["reasoning_tokens"] or 0 for r in ordered_rows)
    warnings_all = [w for r in ordered_rows for w in r.get("warnings", [])]
    empty_count = sum(1 for r in ordered_rows if r["empty_content"])
    truncated_count = sum(1 for r in ordered_rows if r["truncated"])
    overall_correct = sum(1 for r in ordered_rows if r["eval_status"] == "correct")

    summary = {
        "stage": "screen",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": normalize_openai_base_url(base_url),
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "planned_candidates": len(candidates),
        "completed_candidates": len(ordered_rows),
        "requests_sent": requests_sent,
        "empty_content_count": empty_count,
        "unexpected_truncation_count": truncated_count,
        "warning_count": len(warnings_all),
        "warnings_sample": warnings_all[:10],
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "reasoning_tokens": total_reasoning,
        "total_tokens": total_prompt + total_completion,
        "estimated_usd": str(total_usd),
        "overall_eval_counts": dict(Counter(r["eval_status"] for r in ordered_rows)),
        "overall_pass_at_1": (overall_correct / len(ordered_rows)) if ordered_rows else 0.0,
        "by_source": by_source,
        "devset_dir": str(devset_dir),
        "selection": selection_info,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def select_golden_devset(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateProblem],
    *,
    target_count: int = 30,
    min_acc: float = 0.30,
    max_acc: float = 0.65,
) -> tuple[list[CandidateProblem], dict[str, Any]]:
    """Select target_count auto-gradable problems hitting the 30%-65% pass@1 window.

    Counts both wrong final answers (``incorrect``) and 16k-token length exhaustion
    (``no_evaluable_output``) as natural model failures, and prioritizes hard sources
    (``omni_math``, ``olympiadbench``) before saturated ``math500``.
    """
    by_id = {c.problem_id: c for c in candidates}
    pass_rows = [
        r for r in rows
        if r.get("eval_status") == "correct" and is_sympy_auto_evaluable(by_id[r["problem_id"]].reference_answer)
    ]
    fail_rows = [
        r for r in rows
        if r.get("eval_status") in {"incorrect", "no_evaluable_output"}
        and is_sympy_auto_evaluable(by_id[r["problem_id"]].reference_answer)
    ]

    if len(pass_rows) + len(fail_rows) < target_count:
        return [], {
            "status": "insufficient_clean_candidates",
            "pass_count": len(pass_rows),
            "fail_count": len(fail_rows),
        }

    # Target natural pass@1 around ~47% inside [30%, 65%], clamped to available failures/passes.
    min_pass = max(int((target_count * min_acc) + 0.9999), target_count - len(fail_rows))
    max_pass = min(int(target_count * max_acc), len(pass_rows))
    if min_pass > max_pass:
        chosen_pass_n = min(len(pass_rows), max(0, target_count - len(fail_rows)))
    else:
        desired = round(target_count * 0.47)
        chosen_pass_n = max(min_pass, min(max_pass, desired))
    chosen_fail_n = target_count - chosen_pass_n

    def _source_balanced_pick(
        pool: list[Mapping[str, Any]],
        k: int,
        priority: tuple[str, ...],
    ) -> list[Mapping[str, Any]]:
        buckets: dict[str, list[Mapping[str, Any]]] = {
            src: sorted(
                [r for r in pool if r["source"] == src],
                key=lambda r: _stable_hash("track1-dev30-pick", r["problem_id"]),
            )
            for src in priority
        }
        picked: list[Mapping[str, Any]] = []
        # Take from non-saturated hard sources first; only pull math500 if needed.
        hard_sources = [s for s in priority if s != "math500"]
        while len(picked) < k:
            progressed = False
            for src in hard_sources:
                if buckets[src] and len(picked) < k:
                    picked.append(buckets[src].pop(0))
                    progressed = True
            if not progressed:
                break
        while len(picked) < k and buckets.get("math500"):
            picked.append(buckets["math500"].pop(0))
        return picked

    picked_pass = _source_balanced_pick(pass_rows, chosen_pass_n, ("omni_math", "olympiadbench", "math500"))
    picked_fail = _source_balanced_pick(fail_rows, chosen_fail_n, ("omni_math", "olympiadbench", "math500"))
    picked_all = sorted(
        picked_pass + picked_fail,
        key=lambda r: _stable_hash("track1-dev30-order", r["problem_id"]),
    )
    selected_problems = [by_id[r["problem_id"]] for r in picked_all]
    source_counts = dict(Counter(p.source for p in selected_problems))
    fail_breakdown = dict(Counter(r["eval_status"] for r in picked_fail))
    pass_at_1 = len(picked_pass) / len(selected_problems)
    info = {
        "status": "selected" if (min_acc <= pass_at_1 <= max_acc) else "selected_outside_target_band",
        "selected_count": len(selected_problems),
        "baseline_pass_count": len(picked_pass),
        "baseline_fail_count": len(picked_fail),
        "baseline_fail_breakdown": fail_breakdown,
        "baseline_pass_at_1": pass_at_1,
        "by_source": source_counts,
        "selected_problem_ids": [p.problem_id for p in selected_problems],
    }
    return selected_problems, info


def reselect_devset_from_screen(
    screen_dir: Path,
    devset_dir: Path,
    *,
    target_count: int = 30,
) -> dict[str, Any]:
    """Offline re-selection of the 30-problem devset from saved Step 2 screen records."""
    candidates = load_problem_split(screen_dir / "candidates")
    rows = [
        json.loads(line)
        for line in (screen_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected, info = select_golden_devset(rows, candidates, target_count=target_count)
    screen_summary = json.loads((screen_dir / "summary.json").read_text(encoding="utf-8"))
    if selected:
        write_problem_split(
            selected,
            devset_dir,
            metadata={
                "stage": "track1_dev30",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "model": screen_summary.get("model"),
                "max_tokens": screen_summary.get("max_tokens"),
                "temperature": screen_summary.get("temperature"),
                **info,
            },
        )
    screen_summary["selection"] = info
    (screen_dir / "summary.json").write_text(
        json.dumps(screen_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return info


def run_screen_extra_omnimath(
    repo_root: Path,
    screen_dir: Path,
    devset_dir: Path,
    *,
    extra_count: int = 12,
    omnimath_min_difficulty: float = 7.0,
    target_count: int = 30,
    max_tokens: int = 16384,
    temperature: float = 0.6,
    max_requests: int = 14,
    max_estimated_usd: Decimal = Decimal("0.25"),
    max_workers: int = 4,
) -> dict[str, Any]:
    """Screen extra clean Omni-MATH candidates and rebuild the 30-problem golden devset."""
    base_url = os.environ.get("BASE_URL", "")
    model = os.environ.get("MODEL", "")
    api_key = os.environ.get("API_KEY", "")
    if not base_url or not model or not api_key:
        raise RuntimeError("BASE_URL, MODEL, and API_KEY environment variables must be set")

    existing_candidates = load_problem_split(screen_dir / "candidates")
    existing_ids = {c.problem_id for c in existing_candidates}
    existing_groups = {c.question_group_id for c in existing_candidates}
    existing_fps = {problem_fingerprint(c.problem) for c in existing_candidates}

    pools = load_competition_candidates(
        repo_root,
        math500_level=5,
        omnimath_min_difficulty=omnimath_min_difficulty,
    )
    ordered_omni = sorted(
        pools["omni_math"],
        key=lambda item: _stable_hash("track1-exp35-screen-extra-v1", item.problem_id),
    )
    extras: list[CandidateProblem] = []
    for item in ordered_omni:
        fp = problem_fingerprint(item.problem)
        if item.problem_id in existing_ids or item.question_group_id in existing_groups or fp in existing_fps:
            continue
        existing_ids.add(item.problem_id)
        existing_groups.add(item.question_group_id)
        existing_fps.add(fp)
        extras.append(item)
        if len(extras) >= extra_count:
            break

    responses_dir = screen_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    requests_sent = 0
    total_usd = Decimal("0")
    extra_rows: list[dict[str, Any]] = []

    def _run_one(prob: CandidateProblem) -> None:
        nonlocal requests_sent, total_usd
        with lock:
            if requests_sent >= max_requests or total_usd >= max_estimated_usd:
                return
            remaining_attempts = max(1, min(2, max_requests - requests_sent))
            requests_sent += 1

        res = call_chat_completion(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prob.problem + SOLVE_PROMPT_SUFFIX,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            max_attempts=remaining_attempts,
        )
        ev = (
            evaluate_output(res["content"], prob.reference_answer)
            if res["ok"]
            else {
                "status": AnswerStatus.NO_EVALUABLE_OUTPUT.value,
                "extracted_answer": None,
                "normalized_answer": None,
                "reason": res["error"],
            }
        )
        resp_file = responses_dir / f"{prob.problem_id}.json"
        resp_file.write_text(
            json.dumps(
                {
                    "problem_id": prob.problem_id,
                    "source": prob.source,
                    "content": res["content"],
                    "reasoning_content": res["reasoning_content"],
                    "finish_reason": res["finish_reason"],
                    "warnings": res["warnings"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        row = {
            "problem_id": prob.problem_id,
            "source": prob.source,
            "difficulty_tag": prob.difficulty_tag,
            "ok": res["ok"],
            "error": res["error"],
            "attempts_used": res["attempts_used"],
            "wall_time_s": res["wall_time_s"],
            "finish_reason": res["finish_reason"],
            "empty_content": not bool(res["content"].strip()),
            "truncated": res["finish_reason"] != "stop",
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "reasoning_tokens": res["reasoning_tokens"],
            "total_tokens": res["total_tokens"],
            "warnings": res["warnings"],
            "estimated_usd": res["estimated_usd"],
            "eval_status": ev["status"],
            "extracted_answer": ev["extracted_answer"],
            "reference_answer": prob.reference_answer,
            "eval_reason": ev["reason"],
            "response_ref": str(resp_file.relative_to(screen_dir)),
        }
        with lock:
            requests_sent += max(0, res["attempts_used"] - 1)
            total_usd += Decimal(res["estimated_usd"])
            extra_rows.append(row)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [pool.submit(_run_one, prob) for prob in extras]
        for fut in as_completed(futures):
            fut.result()

    all_candidates = existing_candidates + extras
    write_problem_split(
        all_candidates,
        screen_dir / "candidates",
        metadata={"stage": "screen_candidates_augmented", "total_candidates": len(all_candidates)},
    )
    existing_rows = [
        json.loads(line)
        for line in (screen_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_rows = existing_rows + extra_rows
    with (screen_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Exclude known noisy ProcessBench-OlympiadBench labels/TNYWR items from selection.
    noisy_ids = {
        "olympiadbench__olympiadbench-545",
        "olympiadbench__olympiadbench-565",
        "olympiadbench__olympiadbench-687",
    }
    clean_rows = [r for r in all_rows if r["problem_id"] not in noisy_ids]
    selected, info = select_golden_devset(clean_rows, all_candidates, target_count=target_count)
    if selected:
        write_problem_split(
            selected,
            devset_dir,
            metadata={
                "stage": "track1_dev30",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **info,
            },
        )

    total_prompt = sum(r["prompt_tokens"] or 0 for r in all_rows)
    total_completion = sum(r["completion_tokens"] or 0 for r in all_rows)
    total_reasoning = sum(r["reasoning_tokens"] or 0 for r in all_rows)
    all_warnings = [w for r in all_rows for w in r.get("warnings", [])]
    total_all_usd = sum((Decimal(r.get("estimated_usd", "0")) for r in all_rows), Decimal("0"))

    by_source: dict[str, dict[str, Any]] = {}
    for src in ("math500", "omni_math", "olympiadbench"):
        sub = [r for r in all_rows if r["source"] == src]
        correct = sum(1 for r in sub if r["eval_status"] == "correct")
        by_source[src] = {
            "count": len(sub),
            "correct": correct,
            "incorrect": sum(1 for r in sub if r["eval_status"] == "incorrect"),
            "blind_review": sum(1 for r in sub if r["eval_status"] == "blind_review"),
            "no_evaluable_output": sum(1 for r in sub if r["eval_status"] == "no_evaluable_output"),
            "pass_at_1": (correct / len(sub)) if sub else 0.0,
        }

    summary = {
        "stage": "screen",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": normalize_openai_base_url(base_url),
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "planned_candidates": len(all_candidates),
        "completed_candidates": len(all_rows),
        "requests_sent": sum(r.get("attempts_used", 1) for r in all_rows),
        "extra_requests_sent": requests_sent,
        "extra_estimated_usd": str(total_usd),
        "empty_content_count": sum(1 for r in all_rows if r["empty_content"]),
        "unexpected_truncation_count": sum(1 for r in all_rows if r["truncated"]),
        "warning_count": len(all_warnings),
        "warnings_sample": all_warnings[:10],
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "reasoning_tokens": total_reasoning,
        "total_tokens": total_prompt + total_completion,
        "estimated_usd": str(total_all_usd),
        "overall_eval_counts": dict(Counter(r["eval_status"] for r in all_rows)),
        "overall_pass_at_1": sum(1 for r in all_rows if r["eval_status"] == "correct") / len(all_rows),
        "by_source": by_source,
        "devset_dir": str(devset_dir),
        "selection": info,
    }
    (screen_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


DEFAULT_TOKEN_CAPS: tuple[int, ...] = (8192, 17000, 25000, 34000, 50000, 65536)
CONDITION_ORDER: tuple[str, ...] = ("B1", "B1-SC", "B2", "B3", "B4", "RiCL_v0.2")
CONDITION_STYLES: dict[str, dict[str, str]] = {
    "B1": {"label": "B1: Critic BoN", "color": "#4C72B0", "marker": "o", "linestyle": "-"},
    "B1-SC": {"label": "Ref: B1-SC (no-critic vote)", "color": "#7F7F7F", "marker": "x", "linestyle": "--"},
    "B2": {"label": "B2: StepCo Prefix Repair", "color": "#DD8452", "marker": "s", "linestyle": "-"},
    "B3": {"label": "B3: RSE + Same Verifier", "color": "#55A868", "marker": "^", "linestyle": "-"},
    "B4": {"label": "B4: Switch w/o Salvage", "color": "#8172B3", "marker": "v", "linestyle": "-"},
    "RiCL_v0.2": {"label": "Ours: RiCL v0.2", "color": "#C44E52", "marker": "D", "linestyle": "-"},
}


def majority_vote_answers(candidates: Sequence[str | None]) -> str | None:
    """Select the majority answer across rollouts using answer.py equivalence clustering."""
    valid = [ans.strip() for ans in candidates if isinstance(ans, str) and ans.strip()]
    if not valid:
        return None
    clusters: list[dict[str, Any]] = []
    for ans in valid:
        matched = False
        for cluster in clusters:
            rep = cluster["rep"]
            eq = answers_equivalent(rep, ans)
            if eq is True or (eq is None and normalize_answer(rep) == normalize_answer(ans)):
                cluster["count"] += 1
                matched = True
                break
        if not matched:
            clusters.append({"rep": ans, "count": 1})
    # Break ties deterministically by earliest rollout appearance.
    best = max(clusters, key=lambda c: c["count"])
    return str(best["rep"])


def _safe_total_tokens(row: Mapping[str, Any]) -> int:
    total = row.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    pt = row.get("prompt_tokens") or 0
    ct = row.get("completion_tokens") or 0
    return int(pt) + int(ct)


def run_b1(
    devset_dir: Path,
    screen_records_path: Path,
    output_dir: Path,
    *,
    n_rollouts: int = 3,
    max_tokens: int = 16384,
    temperature: float | None = None,
    thinking_type: str = "disabled",
    reasoning_effort: str | None = "none",
    max_requests: int = 75,
    max_estimated_usd: Decimal = Decimal("1.30"),
    max_workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Step 3: Run B1 (Self-Consistency, N=3) on dev30 by reusing Rollout #1 and running N-1 more."""
    if n_rollouts < 2:
        raise ValueError("n_rollouts must be >= 2")
    base_url = os.environ.get("BASE_URL", "")
    model = os.environ.get("MODEL", "")
    api_key = os.environ.get("API_KEY", "")
    if not base_url or not model or not api_key:
        raise RuntimeError("BASE_URL, MODEL, and API_KEY environment variables must be set")

    devset = load_problem_split(devset_dir)
    dev_ids = {p.problem_id for p in devset}

    # Load Rollout #1 from Step 2 screening (same model, prompt, max_tokens, temperature).
    rollout1_by_id: dict[str, dict[str, Any]] = {}
    for line in screen_records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid = row["problem_id"]
        if pid in dev_ids and row.get("ok"):
            rollout1_by_id[pid] = {
                "problem_id": pid,
                "rollout_index": 1,
                "reused_from": "step2_screen",
                "ok": row["ok"],
                "error": row["error"],
                "attempts_used": 0,
                "wall_time_s": row["wall_time_s"],
                "finish_reason": row["finish_reason"],
                "empty_content": row["empty_content"],
                "truncated": row["truncated"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "reasoning_tokens": row["reasoning_tokens"],
                "total_tokens": row["total_tokens"],
                "warnings": row.get("warnings", []),
                "estimated_usd": row["estimated_usd"],
                "extracted_answer": row.get("extracted_answer"),
                "response_ref": row.get("response_ref"),
            }
    if len(rollout1_by_id) != len(devset):
        missing = sorted(dev_ids - set(rollout1_by_id))
        raise RuntimeError(f"Missing Rollout #1 from screen records for: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = output_dir / "rollouts.jsonl"

    existing_fresh: dict[tuple[str, int], dict[str, Any]] = {}
    if resume and rollouts_path.is_file():
        for line in rollouts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("rollout_index", 0) >= 2 and row.get("ok"):
                existing_fresh[(row["problem_id"], int(row["rollout_index"]))] = row

    tasks = [
        (prob, idx)
        for idx in range(2, n_rollouts + 1)
        for prob in devset
        if (prob.problem_id, idx) not in existing_fresh
    ]

    lock = threading.Lock()
    requests_sent = sum(r.get("attempts_used", 1) for r in existing_fresh.values())
    fresh_usd = sum(
        (Decimal(r.get("estimated_usd", "0")) for r in existing_fresh.values()), Decimal("0")
    )
    fresh_rows: dict[tuple[str, int], dict[str, Any]] = dict(existing_fresh)

    def _run_task(prob: CandidateProblem, rollout_idx: int) -> None:
        nonlocal requests_sent, fresh_usd
        with lock:
            if requests_sent >= max_requests or fresh_usd >= max_estimated_usd:
                return
            remaining_attempts = max(1, min(2, max_requests - requests_sent))
            requests_sent += 1

        res = call_chat_completion(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prob.problem + SOLVE_PROMPT_SUFFIX,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            max_attempts=remaining_attempts,
        )
        extracted = extract_final_answer(res["content"]) if res["ok"] and res["content"].strip() else None
        resp_file = responses_dir / f"{prob.problem_id}__r{rollout_idx}.json"
        resp_file.write_text(
            json.dumps(
                {
                    "problem_id": prob.problem_id,
                    "rollout_index": rollout_idx,
                    "source": prob.source,
                    "content": res["content"],
                    "reasoning_content": res["reasoning_content"],
                    "finish_reason": res["finish_reason"],
                    "warnings": res["warnings"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        row = {
            "problem_id": prob.problem_id,
            "rollout_index": rollout_idx,
            "reused_from": None,
            "ok": res["ok"],
            "error": res["error"],
            "attempts_used": res["attempts_used"],
            "wall_time_s": res["wall_time_s"],
            "finish_reason": res["finish_reason"],
            "empty_content": not bool(res["content"].strip()),
            "truncated": res["finish_reason"] != "stop",
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "reasoning_tokens": res["reasoning_tokens"],
            "total_tokens": res["total_tokens"],
            "warnings": res["warnings"],
            "estimated_usd": res["estimated_usd"],
            "extracted_answer": extracted,
            "response_ref": str(resp_file.relative_to(output_dir)),
        }
        with lock:
            requests_sent += max(0, res["attempts_used"] - 1)
            fresh_usd += Decimal(res["estimated_usd"])
            fresh_rows[(prob.problem_id, rollout_idx)] = row
            with rollouts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not resume and rollouts_path.exists():
        rollouts_path.unlink()
        for prob in devset:
            with rollouts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rollout1_by_id[prob.problem_id], ensure_ascii=False) + "\n")
    elif not rollouts_path.exists():
        for prob in devset:
            with rollouts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rollout1_by_id[prob.problem_id], ensure_ascii=False) + "\n")

    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = [pool.submit(_run_task, prob, idx) for prob, idx in tasks]
            for fut in as_completed(futures):
                fut.result()

    all_rollouts: list[dict[str, Any]] = []
    unified_rows: list[dict[str, Any]] = []

    for prob in devset:
        seq = [rollout1_by_id[prob.problem_id]]
        for idx in range(2, n_rollouts + 1):
            if (prob.problem_id, idx) in fresh_rows:
                seq.append(fresh_rows[(prob.problem_id, idx)])
        all_rollouts.extend(seq)

        checkpoints: list[dict[str, Any]] = []
        cum_tokens = 0
        cum_wall = 0.0
        voted_answers: list[str | None] = []
        warnings_all: list[str] = []
        for item in seq:
            cum_tokens += _safe_total_tokens(item)
            cum_wall += float(item.get("wall_time_s") or 0.0)
            voted_answers.append(item.get("extracted_answer"))
            warnings_all.extend(item.get("warnings") or [])
            winner = majority_vote_answers(voted_answers)
            checkpoints.append(
                {
                    "step": item["rollout_index"],
                    "total_tokens": cum_tokens,
                    "final_answer": winner,
                    "independent_eval_ready_text": format_boxed_answer(winner) if winner else None,
                    "rollout_extracted_answer": item.get("extracted_answer"),
                    "finish_reason": item.get("finish_reason"),
                    "ok": item.get("ok", True),
                    "empty_content": item.get("empty_content", False),
                    "error": item.get("error"),
                }
            )
        final_winner = checkpoints[-1]["final_answer"] if checkpoints else None
        unified_rows.append(
            {
                "schema_version": "track1-unified-v0.2",
                "condition": "B1",
                "problem_id": prob.problem_id,
                "source": prob.source,
                "total_tokens": cum_tokens,
                "api_calls": len(seq),
                "fresh_api_calls": sum(1 for item in seq if item["rollout_index"] >= 2),
                "wall_time_s": round(cum_wall, 3),
                "final_answer": final_winner,
                "independent_eval_ready_text": format_boxed_answer(final_winner) if final_winner else None,
                "actions": ["sample"] * len(seq),
                "lemma_stats": {
                    "extracted_count": 0,
                    "l1_verified_count": 0,
                    "pruned_count": 0,
                    "salvaged_reused_count": 0,
                },
                "warnings": warnings_all,
                "checkpoints": checkpoints,
            }
        )

    with rollouts_path.open("w", encoding="utf-8") as fh:
        for row in all_rollouts:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    unified_path = output_dir / "unified_b1.jsonl"
    with unified_path.open("w", encoding="utf-8") as fh:
        for row in unified_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fresh_list = [r for r in all_rollouts if r["rollout_index"] >= 2]
    fresh_prompt = sum(r["prompt_tokens"] or 0 for r in fresh_list)
    fresh_completion = sum(r["completion_tokens"] or 0 for r in fresh_list)
    fresh_reasoning = sum(r["reasoning_tokens"] or 0 for r in fresh_list)
    fresh_warnings = [w for r in fresh_list for w in r.get("warnings", [])]

    all_prompt = sum(r["prompt_tokens"] or 0 for r in all_rollouts)
    all_completion = sum(r["completion_tokens"] or 0 for r in all_rollouts)
    all_reasoning = sum(r["reasoning_tokens"] or 0 for r in all_rollouts)
    all_usd = sum((Decimal(r.get("estimated_usd", "0")) for r in all_rollouts), Decimal("0"))

    summary = {
        "stage": "b1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": normalize_openai_base_url(base_url),
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n_rollouts": n_rollouts,
        "problems": len(devset),
        "fresh_rollouts_completed": len(fresh_list),
        "fresh_requests_sent": requests_sent,
        "fresh_empty_content_count": sum(1 for r in fresh_list if r["empty_content"]),
        "fresh_truncated_count": sum(1 for r in fresh_list if r["truncated"]),
        "fresh_warning_count": len(fresh_warnings),
        "fresh_warnings_sample": fresh_warnings[:10],
        "fresh_prompt_tokens": fresh_prompt,
        "fresh_completion_tokens": fresh_completion,
        "fresh_reasoning_tokens": fresh_reasoning,
        "fresh_total_tokens": fresh_prompt + fresh_completion,
        "fresh_estimated_usd": str(fresh_usd),
        "combined_total_tokens": all_prompt + all_completion,
        "combined_estimated_usd": str(all_usd),
        "unified_jsonl": str(unified_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _parse_int_or_known(value: Any, fallback: Any = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(fallback, int) and not isinstance(fallback, bool):
        return fallback
    return 0


def normalize_unified_record(raw: Mapping[str, Any], *, default_condition: str | None = None) -> dict[str, Any]:
    """Normalize either unified JSONL records or #272 run.json payloads into one schema."""
    condition = str(raw.get("condition") or default_condition or "").strip()
    if not condition:
        raise ValueError("unified record requires non-empty condition")
    problem_id = str(raw.get("problem_id") or raw.get("id") or "").strip()
    if not problem_id:
        raise ValueError("unified record requires non-empty problem_id")

    known_total = _parse_int_or_known(raw.get("known_total_tokens"), 0)
    raw_total = raw.get("total_tokens", known_total)
    resolved_total = _parse_int_or_known(raw_total, known_total)
    is_failed = raw.get("status") == "failed"

    raw_checkpoints = raw.get("checkpoints")
    checkpoints: list[dict[str, Any]] = []
    if isinstance(raw_checkpoints, list) and raw_checkpoints:
        for idx, cp in enumerate(raw_checkpoints, start=1):
            tokens = cp.get("total_tokens", cp.get("tokens"))
            if tokens is None:
                raise ValueError(f"checkpoint {idx} missing total_tokens/tokens for {problem_id}")
            ans = cp.get("final_answer")
            eval_text = cp.get("independent_eval_ready_text") or (
                format_boxed_answer(ans) if isinstance(ans, str) and ans.strip() else None
            )
            checkpoints.append(
                {
                    "step": int(cp.get("step", idx)),
                    "total_tokens": _parse_int_or_known(tokens, 0),
                    "final_answer": ans,
                    "independent_eval_ready_text": eval_text,
                    "finish_reason": cp.get("finish_reason"),
                    "ok": cp.get("ok", True),
                    "empty_content": cp.get("empty_content"),
                    "error": cp.get("error"),
                }
            )
    else:
        ans = raw.get("final_answer")
        eval_text = raw.get("independent_eval_ready_text") or (
            format_boxed_answer(ans) if isinstance(ans, str) and ans.strip() else None
        )
        checkpoints.append(
            {
                "step": 1,
                "total_tokens": resolved_total,
                "final_answer": ans,
                "independent_eval_ready_text": eval_text,
                "finish_reason": raw.get("finish_reason"),
                "ok": raw.get("ok", not is_failed),
                "empty_content": raw.get("empty_content"),
                "error": raw.get("error") or (raw.get("stop_reason") if is_failed else None),
            }
        )
    checkpoints.sort(key=lambda item: (item["total_tokens"], item["step"]))

    return {
        "schema_version": raw.get("schema_version", "track1-unified-v0.2"),
        "condition": condition,
        "problem_id": problem_id,
        "total_tokens": resolved_total,
        "known_total_tokens": known_total,
        "usage_unknown": raw_total == "unknown" or bool(raw.get("unknown_usage_requests")),
        "api_calls": int(raw.get("api_calls", len(checkpoints))),
        "wall_time_s": float(raw.get("wall_time_s", 0.0)),
        "final_answer": raw.get("final_answer", checkpoints[-1]["final_answer"]),
        "independent_eval_ready_text": raw.get(
            "independent_eval_ready_text", checkpoints[-1]["independent_eval_ready_text"]
        ),
        "actions": list(raw.get("actions") or raw.get("action_sequence") or []),
        "lemma_stats": dict(raw.get("lemma_stats") or {}),
        "warnings": list(raw.get("warnings") or []),
        "checkpoints": checkpoints,
    }


def load_unified_runs(
    inputs: Sequence[Path],
    *,
    default_condition: str | None = None,
) -> list[dict[str, Any]]:
    """Load unified JSONL files or #272 run.json artifacts into normalized records."""
    records: list[dict[str, Any]] = []
    for path in inputs:
        target = Path(path)
        if target.is_dir():
            run_files = sorted(target.rglob("run.json"))
            for rf in run_files:
                payload = json.loads(rf.read_text(encoding="utf-8"))
                records.append(normalize_unified_record(payload, default_condition=default_condition))
            continue
        if target.suffix.lower() == ".json":
            payload = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for item in payload:
                    records.append(normalize_unified_record(item, default_condition=default_condition))
            else:
                records.append(normalize_unified_record(payload, default_condition=default_condition))
            continue
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(
                normalize_unified_record(json.loads(line), default_condition=default_condition)
            )
    return records


def evaluate_curves(
    records: Sequence[Mapping[str, Any]],
    answer_key_path: Path,
    *,
    token_caps: Sequence[int] = DEFAULT_TOKEN_CAPS,
) -> dict[str, Any]:
    """Evaluate Accuracy vs. Total Tokens curves offline against separated answer_key.jsonl."""
    answer_key = {
        row["problem_id"]: row["reference_answer"]
        for line in answer_key_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    problem_ids = sorted(answer_key)
    if not problem_ids:
        raise ValueError("answer_key.jsonl contains no problems")

    by_cond: dict[str, dict[str, Mapping[str, Any]]] = {}
    for rec in records:
        by_cond.setdefault(rec["condition"], {})[rec["problem_id"]] = rec

    ordered_conditions = [c for c in CONDITION_ORDER if c in by_cond] + sorted(
        c for c in by_cond if c not in CONDITION_ORDER
    )

    curve_rows: list[dict[str, Any]] = []
    per_problem_eval: list[dict[str, Any]] = []

    for cond in ordered_conditions:
        cond_records = by_cond[cond]
        for cap in sorted(set(int(c) for c in token_caps)):
            counts = Counter()
            realized_tokens: list[int] = []
            for pid in problem_ids:
                rec = cond_records.get(pid)
                if rec is None:
                    counts["missing_run"] += 1
                    continue
                chosen = next(
                    (cp for cp in reversed(rec["checkpoints"]) if cp["total_tokens"] <= cap),
                    None,
                )
                if chosen is None:
                    counts["over_budget_no_checkpoint"] += 1
                    continue
                realized_tokens.append(int(chosen["total_tokens"]))
                eval_text = chosen.get("independent_eval_ready_text") or ""
                if chosen.get("ok") is False or (
                    not eval_text.strip() and chosen.get("finish_reason") is None and chosen.get("error") is not None
                ) or (not eval_text.strip() and chosen.get("finish_reason") is None and chosen.get("step", 0) >= 2):
                    outcome_bucket = "provider_error"
                    ev_extracted = None
                elif not eval_text.strip() and chosen.get("finish_reason") == "length":
                    outcome_bucket = "technical_truncation"
                    ev_extracted = None
                elif not eval_text.strip() and chosen.get("finish_reason") is None:
                    outcome_bucket = "provider_error"
                    ev_extracted = None
                else:
                    ev = evaluate_output(eval_text, answer_key[pid])
                    outcome_bucket = "math_incorrect" if ev["status"] == "incorrect" else ev["status"]
                    ev_extracted = ev["extracted_answer"]
                counts[outcome_bucket] += 1
                per_problem_eval.append(
                    {
                        "condition": cond,
                        "token_cap": cap,
                        "problem_id": pid,
                        "checkpoint_step": chosen["step"],
                        "checkpoint_tokens": chosen["total_tokens"],
                        "eval_status": outcome_bucket,
                        "extracted_answer": ev_extracted,
                        "reference_answer": answer_key[pid],
                    }
                )
            n_total = len(problem_ids)
            n_correct = counts["correct"]
            curve_rows.append(
                {
                    "condition": cond,
                    "token_cap": cap,
                    "n_problems": n_total,
                    "correct": n_correct,
                    "accuracy": n_correct / n_total,
                    "mean_realized_tokens": (
                        round(sum(realized_tokens) / len(realized_tokens), 1)
                        if realized_tokens
                        else None
                    ),
                    "covered_problems": len(realized_tokens),
                    "counts": dict(counts),
                }
            )

    return {
        "schema_version": "track1-curves-v0.2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "n_problems": len(problem_ids),
        "conditions": ordered_conditions,
        "token_caps": sorted(set(int(c) for c in token_caps)),
        "curves": curve_rows,
        "per_problem_eval": per_problem_eval,
    }


def write_curves_and_plot(
    curves_report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Persist curves.json, curves.csv, and Accuracy vs. Total Tokens PNG/SVG plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "curves.json"
    csv_path = output_dir / "curves.csv"
    png_path = output_dir / "accuracy_vs_total_tokens.png"

    json_path.write_text(
        json.dumps(curves_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "condition",
                "token_cap",
                "n_problems",
                "covered_problems",
                "correct",
                "accuracy",
                "mean_realized_tokens",
                "counts_json",
            ],
        )
        writer.writeheader()
        for row in curves_report["curves"]:
            writer.writerow(
                {
                    "condition": row["condition"],
                    "token_cap": row["token_cap"],
                    "n_problems": row["n_problems"],
                    "covered_problems": row["covered_problems"],
                    "correct": row["correct"],
                    "accuracy": f"{row['accuracy']:.4f}",
                    "mean_realized_tokens": row["mean_realized_tokens"],
                    "counts_json": json.dumps(row["counts"], ensure_ascii=False, sort_keys=True),
                }
            )

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    for cond in curves_report["conditions"]:
        pts = [r for r in curves_report["curves"] if r["condition"] == cond]
        pts.sort(key=lambda r: r["token_cap"])
        style = CONDITION_STYLES.get(
            cond, {"label": cond, "color": "#333333", "marker": "o"}
        )
        xs = [r["token_cap"] / 1000.0 for r in pts]
        ys = [r["accuracy"] * 100.0 for r in pts]
        ax.plot(
            xs,
            ys,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style.get("linestyle", "-"),
            linewidth=2.0,
            markersize=6.5,
        )
        for r in pts:
            ax.annotate(
                f"{r['accuracy']*100:.1f}%",
                (r["token_cap"] / 1000.0, r["accuracy"] * 100.0),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=style["color"],
            )

    ax.set_title(
        f"Track 1 Dev ({curves_report['n_problems']} Problems): Accuracy vs. Total Token Budget"
    )
    ax.set_xlabel("Per-Problem Total Token Budget (k tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {"json": json_path, "csv": csv_path, "png": png_path}


def _classify_screen_outcome(row: Mapping[str, Any]) -> str:
    """Distinguish math_correct, math_wrong, technical_truncation, provider_error, and blind_review."""
    if not row.get("ok"):
        return "provider_error"
    if row.get("empty_content") or row.get("finish_reason") == "length":
        return "technical_truncation"
    st = row.get("eval_status")
    if st == "correct":
        return "math_correct"
    if st == "incorrect":
        return "math_wrong"
    if st == "no_evaluable_output":
        return "technical_truncation"
    return "blind_review"


def sample_dev30_v2_candidates(
    repo_root: Path,
    v1_devset_dir: Path,
    *,
    omni_mid_count: int = 28,
    olympiad_extra_count: int = 12,
    math500_count: int = 8,
) -> list[CandidateProblem]:
    """Build dev30-v2 candidate pool: keep stable v1 (<diff 8.0), drop diff>=8.0, add Omni 5.0-6.5 + Oly + MATH-5."""
    v1_probs = load_problem_split(v1_devset_dir)
    kept_v1: list[CandidateProblem] = []
    for p in v1_probs:
        if p.difficulty_tag.startswith("diff_"):
            try:
                if float(p.difficulty_tag.split("_", 1)[1]) >= 8.0:
                    continue
            except ValueError:
                pass
        kept_v1.append(p)

    # Load full pools (Omni-MATH diff >= 5.0, OlympiadBench clean, MATH-500 L5)
    pools = load_competition_candidates(
        repo_root, math500_level=5, omnimath_min_difficulty=5.0
    )
    noisy_ids = {
        "olympiadbench__olympiadbench-545",
        "olympiadbench__olympiadbench-565",
        "olympiadbench__olympiadbench-687",
    }

    selected: list[CandidateProblem] = list(kept_v1)
    seen_ids = {p.problem_id for p in selected}
    seen_groups = {p.question_group_id for p in selected}
    seen_fps = {problem_fingerprint(p.problem) for p in selected}

    def _add_from(pool: Iterable[CandidateProblem], limit: int, seed_tag: str) -> int:
        added = 0
        ordered = sorted(pool, key=lambda item: _stable_hash(seed_tag, item.problem_id))
        for item in ordered:
            if item.problem_id in noisy_ids or item.problem_id in seen_ids:
                continue
            fp = problem_fingerprint(item.problem)
            if item.question_group_id in seen_groups or fp in seen_fps:
                continue
            seen_ids.add(item.problem_id)
            seen_groups.add(item.question_group_id)
            seen_fps.add(fp)
            selected.append(item)
            added += 1
            if added >= limit:
                break
        return added

    omni_mid_pool = [
        p for p in pools["omni_math"]
        if p.difficulty_tag.startswith("diff_")
        and 5.0 <= float(p.difficulty_tag.split("_", 1)[1]) <= 6.5
    ]
    _add_from(omni_mid_pool, omni_mid_count, "dev30-v2-omni-mid")
    _add_from(pools["olympiadbench"], olympiad_extra_count, "dev30-v2-oly")
    _add_from(pools["math500"], math500_count, "dev30-v2-math500")
    return selected


def select_dev30_v2(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateProblem],
    *,
    target_count: int = 30,
    min_wrong: int = 12,
    max_wrong: int = 18,
    max_truncation: int = 3,
) -> tuple[list[CandidateProblem], dict[str, Any]]:
    """Select 30 problems with 12-18 genuine math_wrong (pass@1 ~35%-60%) and <=3 truncations."""
    by_id = {c.problem_id: c for c in candidates}
    correct_rows = [r for r in rows if _classify_screen_outcome(r) == "math_correct"]
    wrong_rows = [r for r in rows if _classify_screen_outcome(r) == "math_wrong"]
    trunc_rows = [r for r in rows if _classify_screen_outcome(r) == "technical_truncation"]

    if len(wrong_rows) < min_wrong:
        # Check if trunc_rows within <= max_truncation could fill, else report shortage honestly.
        needed_trunc = min(max_truncation, max(0, target_count - len(correct_rows) - len(wrong_rows)))
        if len(correct_rows) + len(wrong_rows) + needed_trunc < target_count:
            return [], {
                "status": "insufficient_math_wrong",
                "math_correct_pool": len(correct_rows),
                "math_wrong_pool": len(wrong_rows),
                "technical_truncation_pool": len(trunc_rows),
            }

    # Aim for 15 math_wrong + 15 math_correct (50% pass@1), clamped cleanly into [12, 18].
    desired_wrong = max(min_wrong, min(max_wrong, 15, len(wrong_rows)))
    needed_correct = target_count - desired_wrong
    used_trunc = 0
    if len(correct_rows) < needed_correct:
        # Expand wrong up to max_wrong if needed
        desired_wrong = min(max_wrong, len(wrong_rows), target_count - len(correct_rows))
        needed_correct = min(len(correct_rows), target_count - desired_wrong)
        used_trunc = min(max_truncation, len(trunc_rows), target_count - desired_wrong - needed_correct)

    if desired_wrong + needed_correct + used_trunc < target_count or not (min_wrong <= desired_wrong <= max_wrong):
        return [], {
            "status": "selection_constraints_unmet",
            "math_correct_pool": len(correct_rows),
            "math_wrong_pool": len(wrong_rows),
            "technical_truncation_pool": len(trunc_rows),
            "desired_wrong": desired_wrong,
            "needed_correct": needed_correct,
        }

    def _pick_diverse(pool: list[Mapping[str, Any]], k: int, tag: str) -> list[Mapping[str, Any]]:
        buckets: dict[str, list[Mapping[str, Any]]] = {
            src: sorted(
                [r for r in pool if r["source"] == src],
                key=lambda r: _stable_hash(tag, r["problem_id"]),
            )
            for src in ("omni_math", "olympiadbench", "math500")
        }
        picked: list[Mapping[str, Any]] = []
        while len(picked) < k:
            progressed = False
            for src in ("omni_math", "olympiadbench", "math500"):
                if buckets[src] and len(picked) < k:
                    picked.append(buckets[src].pop(0))
                    progressed = True
            if not progressed:
                break
        return picked

    picked_wrong = _pick_diverse(wrong_rows, desired_wrong, "dev30-v2-pick-wrong")
    picked_correct = _pick_diverse(correct_rows, needed_correct, "dev30-v2-pick-correct")
    picked_trunc = _pick_diverse(trunc_rows, used_trunc, "dev30-v2-pick-trunc")
    picked_all = sorted(
        picked_wrong + picked_correct + picked_trunc,
        key=lambda r: _stable_hash("dev30-v2-order", r["problem_id"]),
    )
    selected_problems = [by_id[r["problem_id"]] for r in picked_all]
    pass_at_1 = len(picked_correct) / len(selected_problems)
    info = {
        "status": "selected",
        "selected_count": len(selected_problems),
        "math_correct_count": len(picked_correct),
        "math_wrong_count": len(picked_wrong),
        "technical_truncation_count": len(picked_trunc),
        "baseline_pass_at_1": pass_at_1,
        "by_source": dict(Counter(p.source for p in selected_problems)),
        "math_wrong_problem_ids": [r["problem_id"] for r in picked_wrong],
        "selected_problem_ids": [p.problem_id for p in selected_problems],
    }
    return selected_problems, info


def run_screen_v2(
    repo_root: Path,
    v1_devset_dir: Path,
    output_dir: Path,
    devset_v2_dir: Path,
    *,
    max_tokens: int = 16384,
    temperature: float | None = None,
    thinking_type: str = "disabled",
    reasoning_effort: str | None = "none",
    max_requests: int = 80,
    max_estimated_usd: Decimal = Decimal("1.50"),
    max_workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Screen dev30-v2 candidates under PR #273 thinking-disabled config and export dev30-v2 + math_wrong list."""
    base_url = os.environ.get("BASE_URL", "")
    model = os.environ.get("MODEL", "")
    api_key = os.environ.get("API_KEY", "")
    if not base_url or not model or not api_key:
        raise RuntimeError("BASE_URL, MODEL, and API_KEY environment variables must be set")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    if resume and (candidates_dir / "problems.jsonl").is_file():
        candidates = load_problem_split(candidates_dir)
    else:
        candidates = sample_dev30_v2_candidates(repo_root, v1_devset_dir)
        write_problem_split(
            candidates,
            candidates_dir,
            metadata={
                "stage": "dev30_v2_candidates",
                "total_candidates": len(candidates),
                "by_source": dict(Counter(c.source for c in candidates)),
            },
        )

    existing_rows: dict[str, dict[str, Any]] = {}
    if resume and records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                existing_rows[row["problem_id"]] = row

    pending = [c for c in candidates if c.problem_id not in existing_rows]
    lock = threading.Lock()
    requests_sent = sum(r.get("attempts_used", 1) for r in existing_rows.values())
    total_usd = sum((Decimal(r.get("estimated_usd", "0")) for r in existing_rows.values()), Decimal("0"))
    rows_by_id: dict[str, dict[str, Any]] = dict(existing_rows)

    def _run_one(prob: CandidateProblem) -> None:
        nonlocal requests_sent, total_usd
        with lock:
            if requests_sent >= max_requests or total_usd >= max_estimated_usd:
                return
            remaining_attempts = max(1, min(2, max_requests - requests_sent))
            requests_sent += 1

        res = call_chat_completion(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prob.problem + SOLVE_PROMPT_SUFFIX,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            max_attempts=remaining_attempts,
        )
        ev = (
            evaluate_output(res["content"], prob.reference_answer)
            if res["ok"]
            else {
                "status": AnswerStatus.NO_EVALUABLE_OUTPUT.value,
                "extracted_answer": None,
                "normalized_answer": None,
                "reason": res["error"],
            }
        )
        resp_file = responses_dir / f"{prob.problem_id}.json"
        resp_file.write_text(
            json.dumps(
                {
                    "problem_id": prob.problem_id,
                    "source": prob.source,
                    "content": res["content"],
                    "reasoning_content": res["reasoning_content"],
                    "finish_reason": res["finish_reason"],
                    "warnings": res["warnings"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        row = {
            "problem_id": prob.problem_id,
            "source": prob.source,
            "difficulty_tag": prob.difficulty_tag,
            "ok": res["ok"],
            "error": res["error"],
            "attempts_used": res["attempts_used"],
            "wall_time_s": res["wall_time_s"],
            "finish_reason": res["finish_reason"],
            "empty_content": not bool(res["content"].strip()),
            "truncated": res["finish_reason"] != "stop",
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "reasoning_tokens": res["reasoning_tokens"],
            "total_tokens": res["total_tokens"],
            "warnings": res["warnings"],
            "estimated_usd": res["estimated_usd"],
            "eval_status": ev["status"],
            "extracted_answer": ev["extracted_answer"],
            "reference_answer": prob.reference_answer,
            "eval_reason": ev["reason"],
            "response_ref": str(resp_file.relative_to(output_dir)),
        }
        row["outcome_bucket"] = _classify_screen_outcome(row)
        with lock:
            requests_sent += max(0, res["attempts_used"] - 1)
            total_usd += Decimal(res["estimated_usd"])
            rows_by_id[prob.problem_id] = row
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not resume and records_path.exists():
        records_path.unlink()

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = [pool.submit(_run_one, prob) for prob in pending]
            for fut in as_completed(futures):
                fut.result()

    ordered_rows = [rows_by_id[c.problem_id] for c in candidates if c.problem_id in rows_by_id]
    with records_path.open("w", encoding="utf-8") as fh:
        for row in ordered_rows:
            row["outcome_bucket"] = _classify_screen_outcome(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected, info = select_dev30_v2(ordered_rows, candidates)
    probe_list_path = devset_v2_dir / "math_wrong_probe_list.json"
    if selected:
        write_problem_split(
            selected,
            devset_v2_dir,
            metadata={
                "stage": "track1_dev30_v2",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "thinking_type": thinking_type,
                "reasoning_effort": reasoning_effort,
                **info,
            },
        )
        rows_idx = {r["problem_id"]: r for r in ordered_rows}
        probs_idx = {p.problem_id: p for p in selected}
        wrong_probe = [
            {
                "problem_id": pid,
                "source": probs_idx[pid].source,
                "difficulty_tag": probs_idx[pid].difficulty_tag,
                "problem": probs_idx[pid].problem,
                "reference_answer": probs_idx[pid].reference_answer,
                "baseline_extracted_answer": rows_idx[pid]["extracted_answer"],
                "baseline_completion_tokens": rows_idx[pid]["completion_tokens"],
                "baseline_finish_reason": rows_idx[pid]["finish_reason"],
                "baseline_response_ref": str(output_dir / rows_idx[pid]["response_ref"]),
            }
            for pid in info["math_wrong_problem_ids"]
        ]
        probe_list_path.write_text(
            json.dumps(wrong_probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    by_source: dict[str, dict[str, Any]] = {}
    for src in ("math500", "omni_math", "olympiadbench"):
        sub = [r for r in ordered_rows if r["source"] == src]
        buckets = Counter(r["outcome_bucket"] for r in sub)
        by_source[src] = {
            "count": len(sub),
            **dict(buckets),
            "pass_at_1": (buckets["math_correct"] / len(sub)) if sub else 0.0,
        }

    total_prompt = sum(r["prompt_tokens"] or 0 for r in ordered_rows)
    total_completion = sum(r["completion_tokens"] or 0 for r in ordered_rows)
    total_reasoning = sum(r["reasoning_tokens"] or 0 for r in ordered_rows)
    all_warnings = [w for r in ordered_rows for w in r.get("warnings", [])]

    summary = {
        "stage": "screen_v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": normalize_openai_base_url(base_url),
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "thinking_type": thinking_type,
        "reasoning_effort": reasoning_effort,
        "planned_candidates": len(candidates),
        "completed_candidates": len(ordered_rows),
        "requests_sent": requests_sent,
        "outcome_counts": dict(Counter(r["outcome_bucket"] for r in ordered_rows)),
        "warning_count": len(all_warnings),
        "warnings_sample": all_warnings[:10],
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "reasoning_tokens": total_reasoning,
        "total_tokens": total_prompt + total_completion,
        "estimated_usd": str(total_usd),
        "by_source": by_source,
        "devset_v2_dir": str(devset_v2_dir),
        "math_wrong_probe_list": str(probe_list_path) if selected else None,
        "selection": info,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary



