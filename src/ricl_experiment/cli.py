"""Unified CLI for starting and explicitly resuming one-sample model runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import resume_run, start_run


def build_parser() -> argparse.ArgumentParser:
    """Construct the supported run/resume command interface."""
    parser = argparse.ArgumentParser(
        prog="python -m ricl_experiment",
        description="RiCL Motivation Study experiment CLI.",
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
        else:
            run_id = resume_run(args.run_id, args.config, args.manifest, args.artifacts_root)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"run_id={run_id}")
    return 0
