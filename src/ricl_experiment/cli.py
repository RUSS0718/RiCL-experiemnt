"""Unified CLI for starting and explicitly resuming one-sample model runs."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .analysis.ms2 import write_ms2_analysis
from .analysis.ms3_repair import analyze_repairs
from .config import load_config, validate_api_key
from .datasets.manifest import (
    export_question_group_exclusions,
    load_sampling_plan,
    load_snapshot,
    prepare_sample_manifest,
    write_overlap_report,
    write_prepared_manifest,
)
from .motivation.ms2 import resume_ms2_run, start_ms2_run
from .motivation.ms3.analysis import write_analysis
from .motivation.ms3.diagnosis import run_diagnosis
from .motivation.ms3_repair import run_ms3_repair
from .runner import resume_run, start_run


def _add_ms2_inputs(parser: argparse.ArgumentParser, *, resume: bool = False) -> None:
    if resume:
        parser.add_argument("--run-id", required=True)
    else:
        parser.add_argument(
            "--phase",
            choices=("pilot", "adjustment_validation", "confirmatory"),
            required=True,
        )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))


def build_parser() -> argparse.ArgumentParser:
    """Construct the supported run/resume command interface."""
    parser = argparse.ArgumentParser(
        prog="python -m ricl_experiment",
        description=(
            "RiCL Motivation Study experiment CLI. prepare reads pinned local JSON snapshots only."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="create a run and dispatch its single sample")
    run.add_argument("--study", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    run.add_argument("--answer-key", type=Path, help="restricted MS2 answer key")
    resume = commands.add_parser("resume", help="resume one named run after fingerprint checks")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--config", required=True, type=Path)
    resume.add_argument("--manifest", required=True, type=Path)
    resume.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    resume.add_argument("--study", choices=("single", "ms2"), default="single")
    resume.add_argument("--answer-key", type=Path, help="restricted MS2 answer key")
    ms2 = commands.add_parser("ms2", help="run a blinded MS2 Checkpoint and rollout study")
    ms2_commands = ms2.add_subparsers(dest="ms2_command", required=True)
    ms2_run = ms2_commands.add_parser("run", help="start three candidates per sample")
    _add_ms2_inputs(ms2_run)
    ms2_resume = ms2_commands.add_parser("resume", help="resume frozen MS2 logical calls")
    _add_ms2_inputs(ms2_resume, resume=True)
    prepare = commands.add_parser(
        "prepare",
        help="freeze from local pinned snapshots without fetching data or calling a model",
    )
    prepare.add_argument(
        "--snapshot",
        required=True,
        action="append",
        type=Path,
        help="local pinned JSON snapshot; repeat for multiple sources",
    )
    prepare.add_argument("--plan", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    export = commands.add_parser(
        "exclude-groups", help="export original-question groups from a frozen manifest"
    )
    export.add_argument("--manifest", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    overlap = commands.add_parser(
        "overlap", help="record shared groups across same-stage MS2/MS3 manifests"
    )
    overlap.add_argument("--left", required=True, type=Path)
    overlap.add_argument("--right", required=True, type=Path)
    overlap.add_argument("--output", required=True, type=Path)
    repair = commands.add_parser(
        "ms3-repair", help="run blinded G0-G3 repairs on a frozen MS3 sample"
    )
    repair.add_argument("--snapshot", required=True, type=Path)
    repair.add_argument("--manifest", required=True, type=Path)
    repair.add_argument("--audit", required=True, type=Path)
    repair.add_argument("--answer-key", required=True, type=Path)
    repair.add_argument("--config", required=True, type=Path)
    repair.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    repair.add_argument("--seed", type=int)
    repair.add_argument("--max-new-tokens", type=int)
    analysis = commands.add_parser(
        "analyze-ms3-repair", help="analyze blinded MS3 evaluations offline"
    )
    analysis.add_argument("--run-dir", required=True, type=Path)
    analysis.add_argument("--evaluations", type=Path)
    analysis.add_argument("--output", type=Path)
    analysis.add_argument("--bootstrap-samples", type=int, default=1000)
    analysis.add_argument("--seed", type=int, default=0)
    diagnose = commands.add_parser(
        "ms3-diagnose", help="run paired, blinded ProcessBench diagnosis calls"
    )
    diagnose.add_argument("--config", required=True, type=Path)
    diagnose.add_argument("--manifest", required=True, type=Path)
    diagnose.add_argument("--output", required=True, type=Path)
    analyze = commands.add_parser(
        "ms3-analyze", help="analyze saved MS3 calls offline against the restricted audit"
    )
    analyze.add_argument("--run", required=True, type=Path)
    analyze.add_argument("--audit", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    ms2_analysis = commands.add_parser(
        "ms2-analyze", help="analyze a saved MS2 trace without model access"
    )
    ms2_analysis.add_argument("--trace", required=True, type=Path)
    ms2_analysis.add_argument("--output", required=True, type=Path)
    ms2_analysis.add_argument("--scoring-dir", type=Path)
    ms2_analysis.add_argument("--bootstrap-samples", type=int, default=10_000)
    ms2_analysis.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one CLI command and report expected operational errors without tracebacks."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.study == "ms2" and args.answer_key is not None:
                run_id, status = start_ms2_run(
                    args.phase, args.config, args.manifest, args.answer_key, args.artifacts_root
                )
            else:
                run_id = start_run(
                    args.study, args.phase, args.config, args.manifest, args.artifacts_root
                )
                status = "completed"
        elif args.command == "resume":
            if args.study == "ms2" and args.answer_key is not None:
                run_id, status = resume_ms2_run(
                    args.run_id, args.config, args.manifest, args.answer_key, args.artifacts_root
                )
            else:
                run_id = resume_run(args.run_id, args.config, args.manifest, args.artifacts_root)
                status = "completed"
        elif args.command == "ms2":
            if args.ms2_command == "run":
                run_id, status = start_ms2_run(
                    args.phase, args.config, args.manifest, args.answer_key, args.artifacts_root
                )
            else:
                run_id, status = resume_ms2_run(
                    args.run_id, args.config, args.manifest, args.answer_key, args.artifacts_root
                )
        elif args.command == "prepare":
            snapshots = [load_snapshot(path) for path in args.snapshot]
            prepared = prepare_sample_manifest(snapshots, load_sampling_plan(args.plan))
            paths = write_prepared_manifest(prepared, args.output)
            print(f"manifest={paths['manifest']}")
            print(f"audit={paths['audit']}")
            print(f"answer_key={paths['answer-key']}")
            print(f"fingerprint={prepared.manifest['fingerprint']}")
            return 0
        elif args.command == "ms3-repair":
            config = load_config(args.config)
            validate_api_key(config)
            run_id = uuid.uuid4().hex
            run_dir = args.artifacts_root.resolve() / "motivation" / run_id
            result = run_ms3_repair(
                config,
                args.snapshot,
                args.manifest,
                args.audit,
                args.answer_key,
                run_dir,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"run_id={run_id}")
            print(f"run_dir={run_dir}")
            print(f"calls={len(result['assignments'])}")
            print(f"usage={json.dumps(result['usage'], sort_keys=True)}")
            return 0
        elif args.command == "analyze-ms3-repair":
            evaluations_path = args.evaluations or args.run_dir / "blind" / "evaluations.jsonl"
            assignments_path = args.run_dir / "restricted" / "assignments.json"
            report = analyze_repairs(
                _read_jsonl(evaluations_path),
                json.loads(assignments_path.read_text(encoding="utf-8")),
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            output_path = args.output or args.run_dir / "metrics" / "ms3-repair.json"
            _write_json(output_path, report)
            print(f"analysis={output_path}")
            print(
                f"comparisons={json.dumps(report['comparisons'], ensure_ascii=False, sort_keys=True)}"
            )
            return 0
        elif args.command == "ms3-diagnose":
            config = load_config(args.config)
            validate_api_key(config)
            run = run_diagnosis(config, args.manifest, args.output)
            print(f"run={args.output / 'run.json'}")
            print(f"status={run['status']}")
            return 0 if run["status"] == "completed" else 2
        elif args.command == "ms3-analyze":
            output_path = write_analysis(args.run, args.audit, args.output)
            print(f"analysis={output_path}")
            return 0
        elif args.command == "ms2-analyze":
            output_path = write_ms2_analysis(
                args.trace,
                args.output,
                scoring_dir=args.scoring_dir,
                bootstrap_replicates=args.bootstrap_samples,
                seed=args.seed,
            )
            print(f"analysis={output_path}")
            print(f"coverage_figure={output_path.with_name(f'{output_path.stem}-coverage.png')}")
            return 0
        elif args.command == "exclude-groups":
            output_path = export_question_group_exclusions(args.manifest, args.output)
            print(f"exclusions={output_path}")
            return 0
        else:
            output_path = write_overlap_report(args.left, args.right, args.output)
            print(f"overlap_report={output_path}")
            return 0
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"run_id={run_id}")
    if status != "completed":
        print(f"status={status}")
        return 1
    return 0


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read nonblank JSONL records for deterministic offline analysis."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row {number} must be an object")
        rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    """Atomically write a UTF-8 JSON analysis artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
