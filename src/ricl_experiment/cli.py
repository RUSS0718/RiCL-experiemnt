"""Unified CLI for starting and explicitly resuming one-sample model runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .datasets.manifest import (
    export_question_group_exclusions,
    load_sampling_plan,
    load_snapshot,
    prepare_sample_manifest,
    write_overlap_report,
    write_prepared_manifest,
)
from .runner import resume_run, start_run


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
    resume = commands.add_parser("resume", help="resume one named run after fingerprint checks")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--config", required=True, type=Path)
    resume.add_argument("--manifest", required=True, type=Path)
    resume.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    prepare = commands.add_parser(
        "prepare", help="freeze from local pinned snapshots without fetching data or calling a model"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one CLI command and report expected operational errors without tracebacks."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_id = start_run(
                args.study, args.phase, args.config, args.manifest, args.artifacts_root
            )
        elif args.command == "resume":
            run_id = resume_run(args.run_id, args.config, args.manifest, args.artifacts_root)
        elif args.command == "prepare":
            snapshots = [load_snapshot(path) for path in args.snapshot]
            prepared = prepare_sample_manifest(snapshots, load_sampling_plan(args.plan))
            paths = write_prepared_manifest(prepared, args.output)
            print(f"manifest={paths['manifest']}")
            print(f"audit={paths['audit']}")
            print(f"answer_key={paths['answer-key']}")
            print(f"fingerprint={prepared.manifest['fingerprint']}")
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
    return 0
