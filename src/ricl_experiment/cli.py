"""Command-line entry point for the RiCL experiment repository."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ricl_experiment",
        description="RiCL Motivation Study experiment CLI.",
        epilog="Bootstrap only: study commands are not yet implemented.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
