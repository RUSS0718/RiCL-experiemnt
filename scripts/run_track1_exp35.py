"""CLI entrypoint for Track 1 (RUSS0718/RiCL-experiemnt#35) exploration steps."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from ricl_experiment.track1 import (
    DEFAULT_TOKEN_CAPS,
    evaluate_curves,
    load_unified_runs,
    run_b1,
    run_screen,
    run_screen_extra_omnimath,
    run_screen_v2,
    run_smoke,
    write_curves_and_plot,
)
from ricl_experiment.track1_phase_a import (
    DEFAULT_RUNTIME_DIR,
    run_phase_a1,
    run_phase_a2,
)
from ricl_experiment.track1_phase_c import (
    PHASE_C_CONDITIONS,
    run_phase_c,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    smoke = sub.add_parser("smoke", help="Run Step 1: 3-problem API smoke test")
    smoke.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/track1-exp35/smoke-16k"),
    )
    smoke.add_argument("--max-tokens", type=int, default=16384)
    smoke.add_argument("--temperature", type=float, default=None)
    smoke.add_argument("--thinking-type", type=str, default="disabled")
    smoke.add_argument("--reasoning-effort", type=str, default="none")
    smoke.add_argument("--max-requests", type=int, default=6)
    smoke.add_argument("--max-estimated-usd", type=Decimal, default=Decimal("0.20"))

    screen = sub.add_parser("screen", help="Run Step 2: 72-problem screen & 30-problem devset")
    screen.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/track1-exp35/screen"),
    )
    screen.add_argument(
        "--devset-dir",
        type=Path,
        default=Path("data/track1-dev30"),
    )
    screen.add_argument("--per-source", type=int, default=24)
    screen.add_argument("--omnimath-min-difficulty", type=float, default=7.0)
    screen.add_argument("--target-count", type=int, default=30)
    screen.add_argument("--max-tokens", type=int, default=16384)
    screen.add_argument("--temperature", type=float, default=0.6)
    screen.add_argument("--max-requests", type=int, default=90)
    screen.add_argument("--max-estimated-usd", type=Decimal, default=Decimal("1.00"))
    screen.add_argument("--max-workers", type=int, default=4)
    screen.add_argument("--resume", action="store_true")

    extra = sub.add_parser("screen-extra", help="Screen 12 extra Omni-MATH problems and rebuild dev30")
    extra.add_argument("--screen-dir", type=Path, default=Path("artifacts/track1-exp35/screen"))
    extra.add_argument("--devset-dir", type=Path, default=Path("data/track1-dev30"))
    extra.add_argument("--extra-count", type=int, default=12)
    extra.add_argument("--max-requests", type=int, default=14)
    extra.add_argument("--max-estimated-usd", type=Decimal, default=Decimal("0.25"))
    extra.add_argument("--max-workers", type=int, default=4)

    b1 = sub.add_parser("b1", help="Run Step 3: B1 (N=3 Self-Consistency) on dev30")
    b1.add_argument("--devset-dir", type=Path, default=Path("data/track1-dev30"))
    b1.add_argument(
        "--screen-records",
        type=Path,
        default=Path("artifacts/track1-exp35/screen/records.jsonl"),
    )
    b1.add_argument("--output-dir", type=Path, default=Path("artifacts/track1-exp35/b1"))
    b1.add_argument("--n-rollouts", type=int, default=3)
    b1.add_argument("--max-tokens", type=int, default=16384)
    b1.add_argument("--temperature", type=float, default=None)
    b1.add_argument("--thinking-type", type=str, default="disabled")
    b1.add_argument("--reasoning-effort", type=str, default="none")
    b1.add_argument("--max-requests", type=int, default=75)
    b1.add_argument("--max-estimated-usd", type=Decimal, default=Decimal("1.50"))
    b1.add_argument("--max-workers", type=int, default=4)
    b1.add_argument("--resume", action="store_true")

    screen_v2 = sub.add_parser(
        "screen-v2",
        help="Rebuild dev30-v2 under PR #273 thinking-disabled config",
    )
    screen_v2.add_argument("--v1-devset-dir", type=Path, default=Path("data/track1-dev30"))
    screen_v2.add_argument("--output-dir", type=Path, default=Path("artifacts/track1-exp35/screen-v2"))
    screen_v2.add_argument("--devset-v2-dir", type=Path, default=Path("data/track1-dev30-v2"))
    screen_v2.add_argument("--max-tokens", type=int, default=16384)
    screen_v2.add_argument("--temperature", type=float, default=None)
    screen_v2.add_argument("--thinking-type", type=str, default="disabled")
    screen_v2.add_argument("--reasoning-effort", type=str, default="none")
    screen_v2.add_argument("--max-requests", type=int, default=80)
    screen_v2.add_argument("--max-estimated-usd", type=Decimal, default=Decimal("1.50"))
    screen_v2.add_argument("--max-workers", type=int, default=4)
    screen_v2.add_argument("--resume", action="store_true")

    eval_plot = sub.add_parser(
        "eval-plot",
        help="Offline evaluate unified JSONL / #272 run.json inputs and plot Accuracy vs Total Tokens",
    )
    eval_plot.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Unified JSONL file, run.json file, or directory of #272 run.json artifacts",
    )
    eval_plot.add_argument(
        "--answer-key",
        type=Path,
        default=Path("data/track1-dev30/answer_key.jsonl"),
    )
    eval_plot.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/track1-exp35/eval"),
    )
    eval_plot.add_argument("--default-condition", type=str, default=None)
    eval_plot.add_argument(
        "--token-caps",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOKEN_CAPS),
    )

    pa1 = sub.add_parser("phase-a1", help="Run Phase A1: 3-problem #272 E2E smoke")
    pa1.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    pa1.add_argument("--devset-dir", type=Path, default=Path("data/track1-dev30-v2"))
    pa1.add_argument(
        "--b1-unified",
        type=Path,
        default=Path("artifacts/track1-exp35/b1-v2/unified_b1.jsonl"),
    )
    pa1.add_argument("--output-dir", type=Path, default=Path("artifacts/track1-exp35/phase-a1"))
    pa1.add_argument("--token-limit", type=int, default=49152)
    pa1.add_argument("--request-limit", type=int, default=8)
    pa1.add_argument("--cost-limit-usd", type=str, default="0.25")
    pa1.add_argument("--max-tokens", type=int, default=8192)
    pa1.add_argument("--reasoning-effort", type=str, default="none")

    pa2 = sub.add_parser("phase-a2", help="Run Phase A2: 10-problem 3-arm utility probe")
    pa2.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    pa2.add_argument("--devset-dir", type=Path, default=Path("data/track1-dev30-v2"))
    pa2.add_argument("--output-dir", type=Path, default=Path("artifacts/track1-exp35/phase-a2"))
    pa2.add_argument("--max-requests", type=int, default=65)
    pa2.add_argument("--max-estimated-usd", type=Decimal, default=Decimal("0.90"))
    pa2.add_argument("--max-workers", type=int, default=3)

    pc = sub.add_parser("phase-c", help="Run Phase C: matched 5-baseline comparison via track1_cli")
    pc.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    pc.add_argument("--devset-dir", type=Path, default=Path("data/track1-dev30-v2"))
    pc.add_argument(
        "--b1-sc-unified",
        type=Path,
        default=Path("artifacts/track1-exp35/b1-v2/unified_b1.jsonl"),
    )
    pc.add_argument("--output-dir", type=Path, default=Path("artifacts/track1-exp35/phase-c"))
    pc.add_argument("--base-url", type=str, default=None, help="Defaults to User/process BASE_URL")
    pc.add_argument("--model", type=str, default=None, help="Defaults to User/process MODEL")
    pc.add_argument("--conditions", nargs="+", default=list(PHASE_C_CONDITIONS))
    pc.add_argument("--problem-ids", nargs="+", default=None)
    pc.add_argument("--token-limit", type=int, default=32768)
    pc.add_argument("--request-limit", type=int, default=8)
    pc.add_argument("--cost-limit-usd", type=str, default="0.25")
    pc.add_argument("--max-tokens", type=int, default=8192)
    pc.add_argument("--reasoning-effort", type=str, default="none")
    pc.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Total request ceiling; defaults to planned_runs * request_limit",
    )
    pc.add_argument("--max-estimated-usd", type=Decimal, default=Decimal("5.50"))
    pc.add_argument("--max-workers", type=int, default=4)
    pc.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.stage == "smoke":
        summary = run_smoke(
            repo_root,
            args.output_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking_type=args.thinking_type,
            reasoning_effort=args.reasoning_effort,
            max_requests=args.max_requests,
            max_estimated_usd=args.max_estimated_usd,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "screen":
        summary = run_screen(
            repo_root,
            args.output_dir,
            args.devset_dir,
            per_source=args.per_source,
            omnimath_min_difficulty=args.omnimath_min_difficulty,
            target_count=args.target_count,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_requests=args.max_requests,
            max_estimated_usd=args.max_estimated_usd,
            max_workers=args.max_workers,
            resume=args.resume,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "screen-extra":
        summary = run_screen_extra_omnimath(
            repo_root,
            args.screen_dir,
            args.devset_dir,
            extra_count=args.extra_count,
            max_requests=args.max_requests,
            max_estimated_usd=args.max_estimated_usd,
            max_workers=args.max_workers,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "b1":
        summary = run_b1(
            args.devset_dir,
            args.screen_records,
            args.output_dir,
            n_rollouts=args.n_rollouts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking_type=args.thinking_type,
            reasoning_effort=args.reasoning_effort,
            max_requests=args.max_requests,
            max_estimated_usd=args.max_estimated_usd,
            max_workers=args.max_workers,
            resume=args.resume,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "screen-v2":
        summary = run_screen_v2(
            repo_root,
            args.v1_devset_dir,
            args.output_dir,
            args.devset_v2_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking_type=args.thinking_type,
            reasoning_effort=args.reasoning_effort,
            max_requests=args.max_requests,
            max_estimated_usd=args.max_estimated_usd,
            max_workers=args.max_workers,
            resume=args.resume,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "eval-plot":
        records = load_unified_runs(args.input, default_condition=args.default_condition)
        report = evaluate_curves(records, args.answer_key, token_caps=args.token_caps)
        paths = write_curves_and_plot(report, args.output_dir)
        print(
            json.dumps(
                {
                    "n_problems": report["n_problems"],
                    "conditions": report["conditions"],
                    "curves": report["curves"],
                    "outputs": {k: str(v) for k, v in paths.items()},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.stage == "phase-a1":
        summary = run_phase_a1(
            args.runtime_dir,
            args.devset_dir,
            args.b1_unified,
            args.output_dir,
            token_limit=args.token_limit,
            request_limit=args.request_limit,
            cost_limit_usd=args.cost_limit_usd,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "phase-a2":
        summary = run_phase_a2(
            args.runtime_dir,
            args.devset_dir,
            args.output_dir,
            max_requests=args.max_requests,
            max_estimated_usd=args.max_estimated_usd,
            max_workers=args.max_workers,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "phase-c":
        summary = run_phase_c(
            args.runtime_dir,
            args.devset_dir,
            args.b1_sc_unified,
            args.output_dir,
            base_url=args.base_url,
            model=args.model,
            conditions=args.conditions,
            problem_ids=args.problem_ids,
            token_limit=args.token_limit,
            request_limit=args.request_limit,
            cost_limit_usd=args.cost_limit_usd,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            max_requests=args.max_requests,
            max_estimated_usd=args.max_estimated_usd,
            max_workers=args.max_workers,
            resume=not args.no_resume,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
