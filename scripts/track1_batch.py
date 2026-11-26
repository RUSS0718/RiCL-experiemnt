"""Run serial Track 1 batches through a pinned RiCL Runtime checkout."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

from ricl_experiment.evaluation.answer import classify_answer

CONDITIONS = ("B1", "B2", "B3", "RiCL_v0.2")


def sha256(path: Path) -> str:
    """Fingerprint a frozen input or output file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Publish a complete JSON snapshot atomically."""
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def score(answer: str | None, key: str) -> str:
    """Grade locally using the experiment repository's existing closed-answer rules."""
    return str(classify_answer(f"Final answer: {answer}" if answer else None, key).status)


def summarize(rows: list[dict], caps: list[int]) -> list[dict]:
    """Use the last observed checkpoint within each fixed cap, retaining missingness."""
    summary = []
    for condition in dict.fromkeys(row["condition"] for row in rows):
        group = [row for row in rows if row["condition"] == condition]
        for cap in caps:
            counts = Counter()
            measured_costs = []
            for row in group:
                point = next((p for p in reversed(row.get("checkpoints", [])) if p["tokens"] <= cap), None)
                counts[point["evaluation"] if point else row.get("availability", "no_checkpoint")] += 1
                if point:
                    measured_costs.append(point["tokens"])
            summary.append({
                "condition": condition, "token_cap": cap, "problems": len(group),
                "correct_over_all": counts["correct"] / len(group), "counts": dict(counts),
                "mean_checkpoint_tokens": (
                    sum(measured_costs) / len(measured_costs) if measured_costs else None
                ),
            })
    return summary


def main() -> int:
    """Freeze the queue and send only sanitized problem-file inputs to the Runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("smoke", "screen", "compare"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    runtime = args.runtime.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True).strip()
    experiment_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "ricl_runtime"], cwd=runtime, text=True
    )
    if dirty:
        parser.error("the Runtime checkout must match its recorded revision")
    spec = json.loads(args.budget.read_text(encoding="utf-8-sig"))
    with args.manifest.open(encoding="utf-8-sig", newline="") as file:
        problems = list(csv.DictReader(file))
    keys = [row["id"] for row in problems]
    if not keys or len(set(keys)) != len(keys) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]) or not row["problem"].strip()
        or not row["answer"].strip() for row in problems
    ):
        parser.error("manifest requires unique IDs, public problems and offline reference answers")
    if args.stage == "smoke" and len(problems) != 3:
        parser.error("smoke requires exactly three distinct problems")
    if args.stage == "compare" and len(problems) != 30:
        parser.error("comparison requires thirty distinct problems")
    queue = []
    for offset, problem in enumerate(problems):
        conditions = {
            "smoke": ("RiCL_v0.2",),
            "screen": ("B1",),
            "compare": CONDITIONS[offset % 4:] + CONDITIONS[:offset % 4],
        }[args.stage]
        queue.extend((problem, condition) for condition in conditions)
    if spec["request_limit"] * len(queue) > spec["batch_request_limit"] or (
        Decimal(spec["cost_limit_usd"]) * len(queue) > Decimal(spec["batch_cost_limit_usd"])
    ):
        parser.error("the frozen queue exceeds its batch caps")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", {
        "stage": args.stage, "runtime_revision": revision, "experiment_revision": experiment_revision,
        "manifest_sha256": sha256(args.manifest), "budget_sha256": sha256(args.budget),
        "budget": spec, "queue": [{"id": p["id"], "condition": c} for p, c in queue],
    })
    rows = []
    halted = False
    for index, (problem, condition) in enumerate(queue, 1):
        case = {"id": problem["id"], "condition": condition}
        if halted:
            rows.append({**case, "availability": "unexecuted"})
            continue
        print(f"{index}/{len(queue)} running {case['id']} {condition}", flush=True)
        run_dir = root / f"{index:03d}-{case['id']}-{condition}"
        # The child sees only the problem file, never the reference or manifest path.
        question = root / "active-problem.txt"
        question.write_text(problem["problem"], encoding="utf-8")
        try:
            command = [
                sys.executable, "-m", "ricl_runtime.track1_cli", "--problem-file", str(question),
                "--problem-id", case["id"], "--condition", condition,
                "--source-revision", revision, "--output-dir", str(run_dir),
                "--request-limit", str(spec["request_limit"]),
                "--token-limit", str(spec["token_limit"]),
                "--max-tokens", str(spec["max_tokens"]),
                "--reasoning-effort", spec.get("reasoning_effort", "high"),
                "--cost-limit-usd", spec["cost_limit_usd"],
            ]
            if args.stage == "screen":
                command.append("--screen")
            child = subprocess.run(
                command, cwd=runtime, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        finally:
            question.unlink()
        run_file = run_dir / "run.json"
        if not run_file.is_file():
            rows.append({**case, "availability": "runner_failed", "exit_code": child.returncode})
            halted = True
        else:
            run = json.loads(run_file.read_text(encoding="utf-8"))
            for point in run["checkpoints"]:
                point["evaluation"] = score(point["final_answer"], problem["answer"])
                del point["final_answer"]
            rows.append({
                **case, "status": run["status"], "availability": run["stop_reason"],
                "config_hash": run["config_hash"], "actions": run["action_sequence"],
                "known_total_tokens": run["known_total_tokens"], "total_tokens": run["total_tokens"],
                "calls": run["api_calls"], "usd": run["estimated_or_reserved_usd"],
                "preflight_usd": run["cost_preflight_upper_usd"],
                "actual_paid_usd": run["actual_paid_usd"],
                "checkpoints": run["checkpoints"], "trace_sha256": sha256(run_dir / "trace.jsonl"),
            })
            halted = child.returncode not in (0, 1) or (
                run["status"] == "failed" and run["stop_reason"] != "empty_content"
            )
        print(f"{index}/{len(queue)} settled {rows[-1]['availability']}", flush=True)
        write_json(root / "progress.json", {"settled": index, "planned": len(queue), "last": case})
    caps = sorted(set(spec["report_token_caps"]) | {spec["token_limit"]})
    write_json(root / "summary.json", {
        "rows": rows, "curves": summarize(rows, caps),
        "known_tokens": sum(row.get("known_total_tokens", 0) for row in rows),
        "known_attempts": sum(row.get("calls", 0) for row in rows),
        "estimated_or_reserved_usd": str(sum(Decimal(row.get("usd", "0")) for row in rows)),
        "cost_preflight_upper_usd": str(sum(Decimal(row.get("preflight_usd", "0")) for row in rows)),
        "actual_paid_usd": "unknown", "runner_halted": halted,
    })
    return int(halted)


if __name__ == "__main__":
    raise SystemExit(main())
