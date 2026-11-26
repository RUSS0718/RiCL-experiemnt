"""Phase A runner: A1 (#272 3-problem E2E smoke) and A2 (10-problem 3-arm utility probe)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .track1 import (
    SOLVE_PROMPT_SUFFIX,
    call_chat_completion,
    evaluate_curves,
    evaluate_output,
    format_boxed_answer,
    load_problem_split,
    load_unified_runs,
    write_curves_and_plot,
)

DEFAULT_RUNTIME_DIR = Path(r"D:\project\工程化MathAgent\Challenge-Cup-2026")

A1_PROBLEM_IDS: tuple[str, ...] = (
    "olympiadbench__olympiadbench-753",
    "omnimath__test_01238",
    "omnimath__test_02553",
)

A2_CURATED_LEMMAS: dict[str, tuple[str, str]] = {
    "olympiadbench__olympiadbench-753": (
        "T = 3 and K = T + 2 = 5, so we need the largest 5-digit integer with distinct digits divisible by 63 = 9 * 7.",
        "The maximum digit sum of 5 distinct digits is 9 + 8 + 7 + 6 + 5 = 35, so the digit sum of any valid multiple of 9 must be 27, 18, or 9 (36 is impossible).",
    ),
    "omnimath__test_01238": (
        "Swapping summations gives sum_{gcd(a,b)=1} floor(300/(2a+3b)) = sum_{k>=1} C(floor(300/k)), where C(M) = #{(a,b) in Z_{>0}^2 : gcd(a,b)=1, 2a+3b <= M}.",
        "Letting F(M) = #{(a,b) in Z_{>0}^2 : 2a+3b <= M}, Mobius inversion gives C(M) = sum_{d>=1} mu(d) * F(floor(M/d)).",
    ),
    "omnimath__test_01493": (
        "The numerator satisfies 32016000000000001 = 32016 * 10^12 + 1 = 101 * 316990099009901.",
        "Since 316990099009901 is approximately 3.17 * 10^14 and is a product of two distinct primes, its smaller prime factor is at most sqrt(3.17 * 10^14) ~ 1.78 * 10^7.",
    ),
    "omnimath__test_02553": (
        "Three queens cannot all lie on the same row, column, or diagonal, because the middle queen blocks the two outer queens from attacking each other.",
        "Therefore the three queens must form a non-degenerate triangle whose three sides all lie on chessboard lines (rows, columns, or diagonals, i.e., slopes 0, infinity, +1, -1).",
    ),
    "omnimath__test_00160": (
        "Under vertex-sharing (8-neighbor) same-color adjacency, the maximum number of pairwise disconnected squares equals the maximum possible number of monochromatic connected components of an n x n black-and-white board (with n >= 3 odd).",
        "Diagonally touching squares of the same color belong to the same component, so any isolated single-square component must be surrounded on all 8 neighboring squares by the opposite color.",
    ),
    "omnimath__test_00010": (
        "In graph terms, G has |V| = 30, maximum degree Delta(G) <= 5, and clique number omega(G) <= 4 (no K_5); the target k is the minimum possible independence number alpha(G) over all such graphs.",
        "By the greedy independent set bound, every graph with |V| = 30 and Delta(G) <= 5 has alpha(G) >= ceil(30 / (Delta + 1)) = 5.",
    ),
    "omnimath__test_01045": (
        "Labeling stones 0, 1, ..., 41 (mod 42), we count directed Hamiltonian cycles starting at 0 using edges +1 and +7 (mod 42).",
        "Since gcd(7, 42) = 7, the 42 stones partition into 7 residue classes mod 7 of 6 stones each; +7 moves stay within the current residue class while +1 moves advance the residue class by +1 (mod 7).",
    ),
    "omnimath__test_00009": (
        "The problem asks for the maximum number of vertices of a strictly convex lattice polygon (no three vertices collinear) contained in a 2004 x 2004 integer grid (width and height 2003).",
        "Each side of a strictly convex lattice polygon has a distinct primitive integer direction vector (dx, dy) with gcd(|dx|, |dy|) = 1, summing to (0, 0) with sum |dx| <= 2 * 2003 and sum |dy| <= 2 * 2003.",
    ),
    "omnimath__test_03617": (
        "Inverting the two forward maps a_{i+1} = 2a_i + 1 and a_{i+1} = a_i/(a_i + 2) yields backward transitions a_i = (a_{i+1} - 1)/2 and a_i = 2a_{i+1}/(1 - a_{i+1}) (for a_{i+1} != 1).",
        "Since a_0 is a positive integer, every forward step keeps a_i > 0, so any backward transition yielding a_i <= 0 (such as 2*2014/(1 - 2014) < 0) is unreachable from a_0 > 0.",
    ),
    "omnimath__test_01795": (
        "The problem counts connected simple graphs on 10 equally spaced vertices on a circle up to symmetries of the dihedral group D_10 (order 20: 10 rotations and 10 reflections).",
        "By Burnside's lemma, the count is (1/20) * sum_{g in D_10} c(g), where c(g) is the number of connected labeled graphs on 10 vertices fixed by g.",
    ),
}

A2_ARMS: tuple[str, ...] = ("switch_no_salvage", "rse_same_verifier", "curated_lemmas")


def _import_runtime_verification(runtime_dir: Path):
    """Import #272's L1/JSON verification helpers without modifying the main repo."""
    rt = str(runtime_dir.resolve())
    if rt not in sys.path:
        sys.path.insert(0, rt)
    from ricl_runtime.track1_verification import CHECK_FORMAT, check_assertion, json_object

    return CHECK_FORMAT, check_assertion, json_object


def _classify_outcome(row: Mapping[str, Any]) -> str:
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


def run_phase_a1(
    runtime_dir: Path,
    devset_dir: Path,
    b1_unified_path: Path,
    output_dir: Path,
    *,
    problem_ids: Sequence[str] = A1_PROBLEM_IDS,
    token_limit: int = 49152,
    request_limit: int = 8,
    cost_limit_usd: str = "0.25",
    max_tokens: int = 8192,
    reasoning_effort: str = "none",
) -> dict[str, Any]:
    """Phase A1: Run 3 dev30-v2 math_wrong problems through merged #272 track1_cli and eval-plot."""
    runtime_dir = runtime_dir.resolve()
    output_dir = output_dir.resolve()
    b1_unified_path = b1_unified_path.resolve()
    rev = subprocess.check_output(
        ["git", "-C", str(runtime_dir), "rev-parse", "HEAD"], text=True
    ).strip()

    problems = {p.problem_id: p for p in load_problem_split(devset_dir)}
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{runtime_dir}{os.pathsep}{existing_pp}" if existing_pp else str(runtime_dir)

    rows: list[dict[str, Any]] = []
    for idx, pid in enumerate(problem_ids, start=1):
        prob = problems[pid]
        run_dir = runs_root / f"{idx:02d}-{pid}"
        run_file = run_dir / "run.json"
        trace_file = run_dir / "trace.jsonl"
        proc_returncode = 0
        proc_stderr = ""
        if not run_file.is_file():
            tmp_problem = output_dir / "_active_problem.txt"
            tmp_problem.write_text(prob.problem, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ricl_runtime.track1_cli",
                        "--problem-file",
                        str(tmp_problem),
                        "--problem-id",
                        pid,
                        "--condition",
                        "RiCL_v0.2",
                        "--source-revision",
                        rev,
                        "--output-dir",
                        str(run_dir),
                        "--token-limit",
                        str(token_limit),
                        "--request-limit",
                        str(request_limit),
                        "--cost-limit-usd",
                        str(cost_limit_usd),
                        "--max-tokens",
                        str(max_tokens),
                        "--reasoning-effort",
                        reasoning_effort,
                    ],
                    cwd=str(runtime_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                proc_returncode = proc.returncode
                proc_stderr = proc.stderr
            finally:
                if tmp_problem.exists():
                    tmp_problem.unlink()

        if not run_file.is_file():
            rows.append(
                {
                    "problem_id": pid,
                    "exit_code": proc_returncode,
                    "status": "runner_failed",
                    "stderr": proc_stderr[-500:],
                }
            )
            continue

        run_data = json.loads(run_file.read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in trace_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        verifications = [e for e in events if e.get("event") == "verification"]
        graphs = [e for e in events if e.get("event") == "lemma_graph"]
        salvage_dispatches = [
            e
            for e in events
            if e.get("event") == "request_started"
            and e.get("stage") == "solve"
            and bool(e.get("lemmas_supplied"))
        ]
        citations = [e for e in events if e.get("event") == "lemma_reuse"]
        final_ans = run_data.get("final_answer")
        ev = evaluate_output(
            format_boxed_answer(final_ans) if isinstance(final_ans, str) and final_ans.strip() else "",
            prob.reference_answer,
        )
        rows.append(
            {
                "problem_id": pid,
                "exit_code": proc_returncode,
                "status": run_data.get("status"),
                "stop_reason": run_data.get("stop_reason"),
                "api_calls": run_data.get("api_calls"),
                "known_total_tokens": run_data.get("known_total_tokens"),
                "total_tokens": run_data.get("total_tokens"),
                "estimated_or_reserved_usd": run_data.get("estimated_or_reserved_usd"),
                "cost_preflight_upper_usd": run_data.get("cost_preflight_upper_usd"),
                "action_sequence": run_data.get("action_sequence"),
                "warnings": run_data.get("warnings"),
                "final_answer": final_ans,
                "reference_answer": prob.reference_answer,
                "eval_status": ev["status"],
                "verification_events": verifications,
                "lemma_graph_events": graphs,
                "nonempty_salvage_dispatches": len(salvage_dispatches),
                "lemmas_supplied_max": max(
                    (len(e.get("lemmas_supplied") or []) for e in salvage_dispatches), default=0
                ),
                "cited_events": citations,
                "run_json": str(run_file),
            }
        )

    # Verify eval-plot integration on the 3-problem subset (B1 vs #272 RiCL_v0.2)
    eval_dir = output_dir / "eval_3prob"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ak3_path = eval_dir / "answer_key_3prob.jsonl"
    with ak3_path.open("w", encoding="utf-8") as fh:
        for pid in problem_ids:
            fh.write(
                json.dumps(
                    {"problem_id": pid, "reference_answer": problems[pid].reference_answer},
                    ensure_ascii=False,
                )
                + "\n"
            )
    unified_recs = [
        rec
        for rec in load_unified_runs([b1_unified_path, runs_root])
        if rec["problem_id"] in set(problem_ids)
    ]
    curves_report = evaluate_curves(
        unified_recs,
        ak3_path,
        token_caps=[2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152],
    )
    plot_paths = write_curves_and_plot(curves_report, eval_dir)

    summary = {
        "stage": "phase_a1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_dir": str(runtime_dir),
        "runtime_revision": rev,
        "problem_ids": list(problem_ids),
        "total_api_calls": sum(int(r.get("api_calls") or 0) for r in rows),
        "total_known_tokens": sum(int(r.get("known_total_tokens") or 0) for r in rows),
        "total_estimated_usd": str(
            sum(
                (Decimal(str(r.get("estimated_or_reserved_usd") or "0")) for r in rows),
                Decimal("0"),
            )
        ),
        "nonempty_salvage_problems": sum(
            1 for r in rows if (r.get("nonempty_salvage_dispatches") or 0) > 0
        ),
        "eval_3prob_conditions": curves_report["conditions"],
        "eval_3prob_outputs": {k: str(v) for k, v in plot_paths.items()},
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_phase_a2(
    runtime_dir: Path,
    devset_dir: Path,
    output_dir: Path,
    *,
    max_requests: int = 65,
    max_estimated_usd: Decimal = Decimal("0.90"),
    max_workers: int = 3,
    max_tokens: int = 16384,
) -> dict[str, Any]:
    """Phase A2: 10-problem 3-arm lightweight utility probe from real math_wrong checkpoints."""
    base_url = os.environ.get("BASE_URL", "")
    model = os.environ.get("MODEL", "")
    api_key = os.environ.get("API_KEY", "")
    if not base_url or not model or not api_key:
        raise RuntimeError("BASE_URL, MODEL, and API_KEY environment variables must be set")

    CHECK_FORMAT, check_assertion, json_object = _import_runtime_verification(runtime_dir)

    probe_items = {
        item["problem_id"]: item
        for item in json.loads((devset_dir / "math_wrong_probe_list.json").read_text(encoding="utf-8"))
    }
    answer_key = {
        row["problem_id"]: row["reference_answer"]
        for line in (devset_dir / "answer_key.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    target_ids = list(A2_CURATED_LEMMAS.keys())
    for pid in target_ids:
        if pid not in probe_items or pid not in answer_key:
            raise KeyError(f"Missing {pid} in math_wrong_probe_list.json or answer_key.jsonl")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    if records_path.exists():
        records_path.unlink()

    lock = threading.Lock()
    requests_sent = 0
    total_usd = Decimal("0")
    records: list[dict[str, Any]] = []

    def _reserve_call(max_tries: int = 2) -> int | None:
        nonlocal requests_sent
        with lock:
            if requests_sent >= max_requests or total_usd >= max_estimated_usd:
                return None
            allowed = max(1, min(max_tries, max_requests - requests_sent))
            requests_sent += 1
            return allowed

    def _settle_call(res: Mapping[str, Any]) -> None:
        nonlocal requests_sent, total_usd
        with lock:
            requests_sent += max(0, int(res["attempts_used"]) - 1)
            total_usd += Decimal(str(res["estimated_usd"]))

    def _run_problem(pid: str) -> list[dict[str, Any]]:
        item = probe_items[pid]
        problem = item["problem"]
        ref_answer = answer_key[pid]
        failed_text = json.loads(
            Path(item["baseline_response_ref"]).read_text(encoding="utf-8")
        )["content"]
        # Bound failed trajectory excerpt to keep extract prompt lean and deterministic.
        failed_excerpt = failed_text[:10000]

        results_for_pid: list[dict[str, Any]] = []

        # Arm 1: Switch without Salvage
        allowed = _reserve_call(2)
        if allowed is not None:
            p_switch = (
                "An earlier attempt to solve this problem using one approach reached an incorrect final answer "
                f"({item['baseline_extracted_answer']}). Discard that failed route completely and do not carry over "
                "any intermediate claims from it. Start a fresh, mathematically distinct route directly from the "
                f"original problem assumptions.{SOLVE_PROMPT_SUFFIX}\n\nProblem:\n{problem}"
            )
            res = call_chat_completion(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=p_switch,
                max_tokens=max_tokens,
                temperature=None,
                thinking_type="disabled",
                reasoning_effort="none",
                max_attempts=allowed,
            )
            _settle_call(res)
            ev = (
                evaluate_output(res["content"], ref_answer)
                if res["ok"]
                else {"status": "no_evaluable_output", "extracted_answer": None, "reason": res["error"]}
            )
            (responses_dir / f"{pid}__switch_no_salvage.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            row = {
                "problem_id": pid,
                "arm": "switch_no_salvage",
                "ok": res["ok"],
                "error": res["error"],
                "api_calls": 1,
                "attempts_used": res["attempts_used"],
                "finish_reason": res["finish_reason"],
                "empty_content": not bool(res["content"].strip()),
                "prompt_tokens": res["prompt_tokens"] or 0,
                "completion_tokens": res["completion_tokens"] or 0,
                "reasoning_tokens": res["reasoning_tokens"],
                "total_tokens": res["total_tokens"] or 0,
                "estimated_usd": res["estimated_usd"],
                "warnings": res["warnings"],
                "eval_status": ev["status"],
                "extracted_answer": ev["extracted_answer"],
                "reference_answer": ref_answer,
            }
            row["outcome_bucket"] = _classify_outcome(row)
            results_for_pid.append(row)

        # Arm 2: RSE + Same Verifier (flat L1 + L2 filtering of lemmas & negative lessons, no DAG)
        rse_tokens_in = 0
        rse_tokens_out = 0
        rse_usd = Decimal("0")
        rse_warnings: list[str] = []
        rse_attempts = 0
        rse_calls = 0

        pos_lemmas: list[dict[str, Any]] = []
        neg_lessons: list[str] = []
        allowed_ext = _reserve_call(2)
        ext_ok = False
        if allowed_ext is not None:
            p_extract = (
                "Extract reusable intermediate lemmas and negative lessons from this failed attempt without "
                "solving the problem. Output compact JSON only with shape: "
                '{"positive_lemmas":[{"claim":"...", "check":null}], "negative_lessons":["..."]}. '
                "List at most 5 standalone positive lemmas that appear mathematically sound and globally applicable "
                "to the original problem, plus at most 3 concise negative lessons identifying pitfalls to avoid. "
                f"For computable arithmetic/algebraic claims, optionally include a check: {CHECK_FORMAT}\n\n"
                f"Problem:\n{problem}\n\nFailed attempt:\n{failed_excerpt}"
            )
            res_ext = call_chat_completion(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=p_extract,
                max_tokens=4096,
                temperature=None,
                thinking_type="disabled",
                reasoning_effort="none",
                json_mode=True,
                max_attempts=allowed_ext,
            )
            _settle_call(res_ext)
            rse_calls += 1
            rse_attempts += int(res_ext["attempts_used"])
            rse_tokens_in += res_ext["prompt_tokens"] or 0
            rse_tokens_out += res_ext["completion_tokens"] or 0
            rse_usd += Decimal(str(res_ext["estimated_usd"]))
            rse_warnings.extend(res_ext["warnings"])
            if res_ext["ok"] and res_ext["finish_reason"] != "length":
                parsed = json_object(res_ext["content"])
                raw_lemmas = parsed.get("positive_lemmas") or []
                raw_lessons = parsed.get("negative_lessons") or []
                if isinstance(raw_lemmas, list):
                    pos_lemmas = [x for x in raw_lemmas[:5] if isinstance(x, dict) and isinstance(x.get("claim"), str)]
                if isinstance(raw_lessons, list):
                    neg_lessons = [str(x).strip() for x in raw_lessons[:3] if isinstance(x, str) and x.strip()]
                ext_ok = True

        # L1 check (same check_assertion as #272)
        l1_pass_indices: set[int] = set()
        surviving_lemmas: list[tuple[int, str, str]] = []
        for idx_l, lem in enumerate(pos_lemmas):
            chk = lem.get("check")
            l1_res = check_assertion(chk) if isinstance(chk, dict) else "uncheckable"
            if l1_res == "fail":
                continue
            if l1_res == "pass":
                l1_pass_indices.add(idx_l)
            surviving_lemmas.append((idx_l, lem["claim"].strip(), l1_res))

        # L2 critic call (same isolated verifier role as #272)
        admitted_lemmas: list[str] = []
        admitted_lessons: list[str] = []
        if ext_ok and (surviving_lemmas or neg_lessons):
            allowed_ver = _reserve_call(2)
            if allowed_ver is not None:
                cand_block = "\n".join(
                    f"Lemma {i}: [{l1s}] {txt}" for i, txt, l1s in surviving_lemmas
                )
                less_block = "\n".join(
                    f"Lesson {i}: {txt}" for i, txt in enumerate(neg_lessons)
                )
                p_verify = (
                    "Independently audit these candidate lemmas and negative lessons against the original problem. "
                    "Approve only items that are mathematically sound and safe to reuse. Output compact JSON only: "
                    '{"valid_lemma_indices":[0], "valid_lesson_indices":[0]}.\n\n'
                    f"Problem:\n{problem}\n\nCandidate lemmas:\n{cand_block or '(none)'}\n\n"
                    f"Candidate lessons:\n{less_block or '(none)'}"
                )
                res_ver = call_chat_completion(
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    prompt=p_verify,
                    max_tokens=2048,
                    temperature=None,
                    thinking_type="disabled",
                    reasoning_effort="none",
                    json_mode=True,
                    max_attempts=allowed_ver,
                )
                _settle_call(res_ver)
                rse_calls += 1
                rse_attempts += int(res_ver["attempts_used"])
                rse_tokens_in += res_ver["prompt_tokens"] or 0
                rse_tokens_out += res_ver["completion_tokens"] or 0
                rse_usd += Decimal(str(res_ver["estimated_usd"]))
                rse_warnings.extend(res_ver["warnings"])
                vjson = json_object(res_ver["content"]) if res_ver["ok"] and res_ver["finish_reason"] != "length" else {}
                l2_lem = set(vjson.get("valid_lemma_indices") or []) if isinstance(vjson.get("valid_lemma_indices"), list) else set()
                l2_les = set(vjson.get("valid_lesson_indices") or []) if isinstance(vjson.get("valid_lesson_indices"), list) else set()
                for orig_idx, txt, l1s in surviving_lemmas:
                    if orig_idx in l1_pass_indices or orig_idx in l2_lem:
                        tag = "L1-Verified" if l1s == "pass" else "L2-Reviewed"
                        admitted_lemmas.append(f"[{tag}] {txt}")
                for idx_s, txt in enumerate(neg_lessons):
                    if idx_s in l2_les:
                        admitted_lessons.append(txt)

        allowed_solve = _reserve_call(2)
        if allowed_solve is not None:
            lem_sec = "\n".join(f"- {t}" for t in admitted_lemmas) or "- (none verified)"
            les_sec = "\n".join(f"- {t}" for t in admitted_lessons) or "- (none verified)"
            p_rse = (
                "An earlier attempt using a different route failed. Start a fresh, mathematically distinct route "
                "from the original assumptions. You may reuse these verifier-filtered facts and lessons from prior "
                "attempts if helpful:\n"
                f"Verified facts:\n{lem_sec}\n\nNegative lessons to avoid:\n{les_sec}"
                f"{SOLVE_PROMPT_SUFFIX}\n\nProblem:\n{problem}"
            )
            res_solve = call_chat_completion(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=p_rse,
                max_tokens=max_tokens,
                temperature=None,
                thinking_type="disabled",
                reasoning_effort="none",
                max_attempts=allowed_solve,
            )
            _settle_call(res_solve)
            rse_calls += 1
            rse_attempts += int(res_solve["attempts_used"])
            rse_tokens_in += res_solve["prompt_tokens"] or 0
            rse_tokens_out += res_solve["completion_tokens"] or 0
            rse_usd += Decimal(str(res_solve["estimated_usd"]))
            rse_warnings.extend(res_solve["warnings"])
            ev = (
                evaluate_output(res_solve["content"], ref_answer)
                if res_solve["ok"]
                else {"status": "no_evaluable_output", "extracted_answer": None, "reason": res_solve["error"]}
            )
            (responses_dir / f"{pid}__rse_same_verifier.json").write_text(
                json.dumps(
                    {
                        "admitted_lemmas": admitted_lemmas,
                        "admitted_lessons": admitted_lessons,
                        "solve_reply": res_solve,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            row = {
                "problem_id": pid,
                "arm": "rse_same_verifier",
                "ok": res_solve["ok"],
                "error": res_solve["error"],
                "api_calls": rse_calls,
                "attempts_used": rse_attempts,
                "finish_reason": res_solve["finish_reason"],
                "empty_content": not bool(res_solve["content"].strip()),
                "prompt_tokens": rse_tokens_in,
                "completion_tokens": rse_tokens_out,
                "reasoning_tokens": res_solve["reasoning_tokens"],
                "total_tokens": rse_tokens_in + rse_tokens_out,
                "estimated_usd": str(rse_usd),
                "warnings": rse_warnings,
                "admitted_lemmas_count": len(admitted_lemmas),
                "admitted_lessons_count": len(admitted_lessons),
                "eval_status": ev["status"],
                "extracted_answer": ev["extracted_answer"],
                "reference_answer": ref_answer,
            }
            row["outcome_bucket"] = _classify_outcome(row)
            results_for_pid.append(row)

        # Arm 3: Curated 1-2 Key Lemmas (post-hoc utility upper-bound probe)
        allowed_cur = _reserve_call(2)
        if allowed_cur is not None:
            lemmas_text = "\n".join(f"- [C{i}] {t}" for i, t in enumerate(A2_CURATED_LEMMAS[pid], start=1))
            p_cur = (
                "An earlier attempt to solve this problem using one approach reached an incorrect final answer "
                f"({item['baseline_extracted_answer']}). Discard that failed route and start a fresh, mathematically "
                "distinct route from the original assumptions. You may directly reuse these verified intermediate "
                f"lemmas established prior to the failure:\n{lemmas_text}{SOLVE_PROMPT_SUFFIX}\n\nProblem:\n{problem}"
            )
            res_cur = call_chat_completion(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=p_cur,
                max_tokens=max_tokens,
                temperature=None,
                thinking_type="disabled",
                reasoning_effort="none",
                max_attempts=allowed_cur,
            )
            _settle_call(res_cur)
            ev = (
                evaluate_output(res_cur["content"], ref_answer)
                if res_cur["ok"]
                else {"status": "no_evaluable_output", "extracted_answer": None, "reason": res_cur["error"]}
            )
            (responses_dir / f"{pid}__curated_lemmas.json").write_text(
                json.dumps(res_cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            row = {
                "problem_id": pid,
                "arm": "curated_lemmas",
                "ok": res_cur["ok"],
                "error": res_cur["error"],
                "api_calls": 1,
                "attempts_used": res_cur["attempts_used"],
                "finish_reason": res_cur["finish_reason"],
                "empty_content": not bool(res_cur["content"].strip()),
                "prompt_tokens": res_cur["prompt_tokens"] or 0,
                "completion_tokens": res_cur["completion_tokens"] or 0,
                "reasoning_tokens": res_cur["reasoning_tokens"],
                "total_tokens": res_cur["total_tokens"] or 0,
                "estimated_usd": res_cur["estimated_usd"],
                "warnings": res_cur["warnings"],
                "eval_status": ev["status"],
                "extracted_answer": ev["extracted_answer"],
                "reference_answer": ref_answer,
            }
            row["outcome_bucket"] = _classify_outcome(row)
            results_for_pid.append(row)

        with lock:
            records.extend(results_for_pid)
            with records_path.open("a", encoding="utf-8") as fh:
                for r in results_for_pid:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return results_for_pid

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [pool.submit(_run_problem, pid) for pid in target_ids]
        for fut in as_completed(futures):
            fut.result()

    # Order deterministically by problem order then arm order
    arm_rank = {arm: i for i, arm in enumerate(A2_ARMS)}
    pid_rank = {pid: i for i, pid in enumerate(target_ids)}
    records.sort(key=lambda r: (pid_rank[r["problem_id"]], arm_rank[r["arm"]]))
    with records_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_arm: dict[str, dict[str, Any]] = {}
    for arm in A2_ARMS:
        sub = [r for r in records if r["arm"] == arm]
        buckets = Counter(r["outcome_bucket"] for r in sub)
        toks = [r["total_tokens"] for r in sub]
        by_arm[arm] = {
            "n": len(sub),
            "math_correct": buckets["math_correct"],
            "accuracy": (buckets["math_correct"] / len(sub)) if sub else 0.0,
            "outcomes": dict(buckets),
            "total_tokens": sum(toks),
            "mean_total_tokens": round(sum(toks) / len(sub), 1) if sub else 0.0,
            "estimated_usd": str(
                sum((Decimal(str(r["estimated_usd"])) for r in sub), Decimal("0"))
            ),
        }

    # Paired table across 10 problems
    by_pid_arm = {(r["problem_id"], r["arm"]): r for r in records}
    paired_rows: list[dict[str, Any]] = []
    for pid in target_ids:
        entry: dict[str, Any] = {"problem_id": pid, "reference_answer": answer_key[pid]}
        for arm in A2_ARMS:
            r = by_pid_arm.get((pid, arm))
            entry[arm] = (
                {
                    "outcome": r["outcome_bucket"],
                    "extracted": r["extracted_answer"],
                    "tokens": r["total_tokens"],
                }
                if r
                else None
            )
        paired_rows.append(entry)

    def _paired_diff(left_arm: str, right_arm: str) -> dict[str, int]:
        wins = losses = ties = 0
        for pid in target_ids:
            l = by_pid_arm.get((pid, left_arm))
            r = by_pid_arm.get((pid, right_arm))
            if not l or not r:
                continue
            lc = l["outcome_bucket"] == "math_correct"
            rc = r["outcome_bucket"] == "math_correct"
            if lc and not rc:
                wins += 1
            elif rc and not lc:
                losses += 1
            else:
                ties += 1
        return {"left_only_correct": wins, "right_only_correct": losses, "ties": ties, "net_gain": wins - losses}

    cur_vs_switch = _paired_diff("curated_lemmas", "switch_no_salvage")
    rse_vs_switch = _paired_diff("rse_same_verifier", "switch_no_salvage")
    cur_vs_rse = _paired_diff("curated_lemmas", "rse_same_verifier")
    go_signal = "GO_TO_PHASE_B" if cur_vs_switch["net_gain"] >= 2 else "NO_GO_OR_REVISE"

    summary = {
        "stage": "phase_a2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "requests_sent": requests_sent,
        "total_tokens": sum(r["total_tokens"] for r in records),
        "estimated_usd": str(total_usd),
        "by_arm": by_arm,
        "paired_comparisons": {
            "curated_vs_switch_no_salvage": cur_vs_switch,
            "rse_same_verifier_vs_switch_no_salvage": rse_vs_switch,
            "curated_vs_rse_same_verifier": cur_vs_rse,
        },
        "go_no_go": go_signal,
        "paired_rows": paired_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
