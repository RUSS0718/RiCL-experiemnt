"""Phase C runner: matched 5-baseline comparison via merged Challenge-Cup-2026 track1_cli."""

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
    evaluate_curves,
    evaluate_output,
    load_problem_split,
    load_unified_runs,
    write_curves_and_plot,
)

PHASE_C_CONDITIONS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "RiCL_v0.2")
PHASE_C_TOKEN_CAPS: tuple[int, ...] = (2048, 4096, 8192, 12288, 16384, 24576, 32768)


def _read_user_env(name: str, *, required: bool = True) -> str:
    """Read fresh User-level env var on Windows, falling back to process env without logging."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg_key:
                val, _ = winreg.QueryValueEx(reg_key, name)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except OSError:
            pass
    val = os.environ.get(name, "").strip()
    if required and not val:
        raise RuntimeError(f"User or process {name} environment variable is not set")
    return val


def _classify_run(run_data: Mapping[str, Any], ref_answer: str) -> tuple[str, str | None]:
    final_ans = run_data.get("final_answer")
    status = run_data.get("status")
    stop_reason = run_data.get("stop_reason")
    if status == "failed" and not (isinstance(final_ans, str) and final_ans.strip()):
        if stop_reason in {"empty_content_at_output_limit", "empty_content"}:
            return "technical_truncation", None
        return "provider_error", None
    if not isinstance(final_ans, str) or not final_ans.strip():
        return (
            "technical_truncation" if stop_reason in {"token_limit", "empty_content_at_output_limit"} else "no_evaluable_output",
            None,
        )
    ev = evaluate_output(f"\\boxed{{{final_ans}}}", ref_answer)
    bucket = "math_wrong" if ev["status"] == "incorrect" else (
        "math_correct" if ev["status"] == "correct" else ev["status"]
    )
    return bucket, ev["extracted_answer"]


def run_phase_c(
    runtime_dir: Path,
    devset_dir: Path,
    b1_sc_unified_path: Path | None,
    output_dir: Path,
    *,
    base_url: str | None = None,
    model: str | None = None,
    conditions: Sequence[str] = PHASE_C_CONDITIONS,
    problem_ids: Sequence[str] | None = None,
    token_limit: int = 32768,
    request_limit: int = 8,
    cost_limit_usd: str = "0.25",
    max_tokens: int = 8192,
    reasoning_effort: str = "none",
    token_caps: Sequence[int] | None = None,
    max_requests: int = 750,
    max_estimated_usd: Decimal = Decimal("5.50"),
    max_workers: int = 4,
    resume: bool = True,
) -> dict[str, Any]:
    """Run matched Track 1 conditions on dev30-v2 via Challenge-Cup-2026 track1_cli."""
    runtime_dir = runtime_dir.resolve()
    output_dir = output_dir.resolve()
    devset_dir = devset_dir.resolve()
    rev = subprocess.check_output(
        ["git", "-C", str(runtime_dir), "rev-parse", "HEAD"], text=True
    ).strip()

    runtime_py = runtime_dir / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python_exe = str(runtime_py) if runtime_py.is_file() else sys.executable

    resolved_base_url = (base_url or _read_user_env("BASE_URL")).strip()
    resolved_model = (model or _read_user_env("MODEL")).strip()
    api_key = _read_user_env("API_KEY")
    base_env = dict(os.environ)
    existing_pp = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = (
        f"{runtime_dir}{os.pathsep}{existing_pp}" if existing_pp else str(runtime_dir)
    )
    base_env["BASE_URL"] = resolved_base_url
    base_env["MODEL"] = resolved_model
    base_env["API_KEY"] = api_key

    all_problems = {p.problem_id: p for p in load_problem_split(devset_dir)}
    selected_ids = list(problem_ids) if problem_ids else list(all_problems.keys())
    for pid in selected_ids:
        if pid not in all_problems:
            raise KeyError(f"Unknown problem_id in devset: {pid}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    queue: list[tuple[str, str]] = [
        (pid, cond) for cond in conditions for pid in selected_ids
    ]

    lock = threading.Lock()
    requests_used = 0
    usd_used = Decimal("0")
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def _summarize_run(pid: str, cond: str, run_dir: Path, exit_code: int, stderr: str = "") -> dict[str, Any]:
        run_file = run_dir / "run.json"
        trace_file = run_dir / "trace.jsonl"
        prob = all_problems[pid]
        if not run_file.is_file():
            return {
                "problem_id": pid,
                "condition": cond,
                "status": "runner_failed",
                "stop_reason": "missing_run_json",
                "exit_code": exit_code,
                "api_calls": 0,
                "known_total_tokens": 0,
                "estimated_or_reserved_usd": "0",
                "outcome_bucket": "provider_error",
                "extracted_answer": None,
                "reference_answer": prob.reference_answer,
                "action_sequence": [],
                "nonempty_salvage_dispatches": 0,
                "warnings": [],
                "stderr_tail": stderr[-300:],
            }
        run_data = json.loads(run_file.read_text(encoding="utf-8"))
        events = (
            [json.loads(l) for l in trace_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if trace_file.is_file()
            else []
        )
        nonempty_salvage = sum(
            1
            for e in events
            if e.get("event") == "request_started"
            and e.get("stage") == "solve"
            and bool(e.get("lemmas_supplied"))
        )
        cited_count = sum(len(e.get("cited") or []) for e in events if e.get("event") == "lemma_reuse")
        bucket, ext = _classify_run(run_data, prob.reference_answer)
        return {
            "problem_id": pid,
            "condition": cond,
            "status": run_data.get("status"),
            "stop_reason": run_data.get("stop_reason"),
            "exit_code": exit_code,
            "api_calls": int(run_data.get("api_calls") or 0),
            "known_total_tokens": int(run_data.get("known_total_tokens") or 0),
            "total_tokens": run_data.get("total_tokens"),
            "estimated_or_reserved_usd": str(run_data.get("estimated_or_reserved_usd") or "0"),
            "cost_preflight_upper_usd": str(run_data.get("cost_preflight_upper_usd") or "0"),
            "wall_time_s": float(run_data.get("wall_time_s") or 0.0),
            "outcome_bucket": bucket,
            "final_answer": run_data.get("final_answer"),
            "extracted_answer": ext,
            "reference_answer": prob.reference_answer,
            "action_sequence": list(run_data.get("action_sequence") or []),
            "nonempty_salvage_dispatches": nonempty_salvage,
            "cited_lemmas_count": cited_count,
            "warnings": list(run_data.get("warnings") or []),
            "run_json": str(run_file),
        }

    pending: list[tuple[str, str]] = []
    for pid, cond in queue:
        run_dir = runs_root / cond / pid
        if resume and (run_dir / "run.json").is_file():
            rec = _summarize_run(pid, cond, run_dir, 0)
            records_by_key[(pid, cond)] = rec
            requests_used += rec["api_calls"]
            usd_used += Decimal(rec["estimated_or_reserved_usd"])
        else:
            pending.append((pid, cond))

    def _run_one(pid: str, cond: str) -> None:
        nonlocal requests_used, usd_used
        with lock:
            if requests_used >= max_requests or usd_used >= max_estimated_usd:
                return
            per_run_req = min(request_limit, max_requests - requests_used)
            requests_used += per_run_req  # Reserve worst-case; reconcile after completion

        run_dir = runs_root / cond / pid
        if run_dir.exists():
            # Clean incomplete directory lacking run.json
            for child in run_dir.iterdir():
                child.unlink()
            run_dir.rmdir()
        run_dir.parent.mkdir(parents=True, exist_ok=True)

        prob = all_problems[pid]
        tmp_problem = output_dir / f"_tmp_{cond}_{pid}.txt"
        tmp_problem.write_text(prob.problem, encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    python_exe,
                    "-m",
                    "ricl_runtime.track1_cli",
                    "--problem-file",
                    str(tmp_problem),
                    "--problem-id",
                    pid,
                    "--condition",
                    cond,
                    "--source-revision",
                    rev,
                    "--output-dir",
                    str(run_dir),
                    "--token-limit",
                    str(token_limit),
                    "--request-limit",
                    str(per_run_req),
                    "--cost-limit-usd",
                    str(cost_limit_usd),
                    "--max-tokens",
                    str(max_tokens),
                    "--reasoning-effort",
                    reasoning_effort,
                ],
                cwd=str(runtime_dir),
                env=base_env,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            if tmp_problem.exists():
                tmp_problem.unlink()

        rec = _summarize_run(pid, cond, run_dir, proc.returncode, proc.stderr)
        with lock:
            requests_used = requests_used - per_run_req + rec["api_calls"]
            usd_used += Decimal(rec["estimated_or_reserved_usd"])
            records_by_key[(pid, cond)] = rec
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not resume and records_path.exists():
        records_path.unlink()

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = [pool.submit(_run_one, pid, cond) for pid, cond in pending]
            for fut in as_completed(futures):
                fut.result()

    ordered_records = [records_by_key[k] for k in queue if k in records_by_key]
    with records_path.open("w", encoding="utf-8") as fh:
        for rec in ordered_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Offline curve evaluation + plot across runtime runs (+ optional B1-SC reference)
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_ak_path = eval_dir / "answer_key_subset.jsonl"
    with eval_ak_path.open("w", encoding="utf-8") as fh:
        for pid in selected_ids:
            fh.write(
                json.dumps(
                    {"problem_id": pid, "reference_answer": all_problems[pid].reference_answer},
                    ensure_ascii=False,
                )
                + "\n"
            )

    unified_records = [
        rec
        for rec in load_unified_runs([runs_root])
        if rec["problem_id"] in set(selected_ids) and rec["condition"] in set(conditions)
    ]
    if b1_sc_unified_path is not None and b1_sc_unified_path.is_file() and len(selected_ids) == len(all_problems):
        for rec in load_unified_runs([b1_sc_unified_path]):
            rec_copy = dict(rec)
            rec_copy["condition"] = "B1-SC"
            unified_records.append(rec_copy)

    effective_caps = (
        list(token_caps)
        if token_caps is not None
        else (list(PHASE_C_TOKEN_CAPS) + ([token_limit] if token_limit > PHASE_C_TOKEN_CAPS[-1] else []))
    )
    curves_report = evaluate_curves(
        unified_records,
        eval_ak_path,
        token_caps=effective_caps,
    )
    plot_paths = write_curves_and_plot(curves_report, eval_dir)

    by_condition: dict[str, dict[str, Any]] = {}
    for cond in conditions:
        sub = [r for r in ordered_records if r["condition"] == cond]
        buckets = Counter(r["outcome_bucket"] for r in sub)
        statuses = Counter(r["status"] for r in sub)
        actions = Counter(a for r in sub for a in r.get("action_sequence", []))
        warnings_count = sum(len(r.get("warnings") or []) for r in sub)
        by_condition[cond] = {
            "completed_runs": len(sub),
            "math_correct": buckets["math_correct"],
            "accuracy": (buckets["math_correct"] / len(selected_ids)) if selected_ids else 0.0,
            "outcomes": dict(buckets),
            "statuses": dict(statuses),
            "api_calls": sum(r["api_calls"] for r in sub),
            "known_total_tokens": sum(r["known_total_tokens"] for r in sub),
            "estimated_usd": str(
                sum((Decimal(r["estimated_or_reserved_usd"]) for r in sub), Decimal("0"))
            ),
            "nonempty_salvage_runs": sum(1 for r in sub if r["nonempty_salvage_dispatches"] > 0),
            "total_cited_lemmas": sum(r.get("cited_lemmas_count", 0) for r in sub),
            "warning_count": warnings_count,
            "action_counts": dict(actions),
        }

    summary = {
        "stage": "phase_c",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_dir": str(runtime_dir),
        "runtime_revision": rev,
        "endpoint": resolved_base_url,
        "model": resolved_model,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "token_limit": token_limit,
        "token_caps": effective_caps,
        "request_limit": request_limit,
        "cost_limit_usd": cost_limit_usd,
        "n_problems": len(selected_ids),
        "conditions": list(conditions),
        "planned_runs": len(queue),
        "executed_runs": len(ordered_records),
        "total_api_calls": sum(r["api_calls"] for r in ordered_records),
        "total_known_tokens": sum(r["known_total_tokens"] for r in ordered_records),
        "total_estimated_usd": str(
            sum((Decimal(r["estimated_or_reserved_usd"]) for r in ordered_records), Decimal("0"))
        ),
        "by_condition": by_condition,
        "eval_conditions": curves_report["conditions"],
        "eval_outputs": {k: str(v) for k, v in plot_paths.items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
