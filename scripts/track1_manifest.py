"""Freeze a licensed BeyondAIME screen and a balanced exploratory comparison set."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from pyarrow import parquet

SOURCE_SHA256 = "969750458ae4796e783b6b5ae004cf48587f10683bd5b5ac00220c50561e1df6"


def order(problem_id: str) -> str:
    """Use a fixed ordering independent of answer or observed model performance."""
    return hashlib.sha256(f"ricl-track1-v02:20261126:{problem_id}".encode()).hexdigest()


def screen_rows(source: Path) -> list[dict]:
    """Choose 48 independent rows deterministically from the pinned original release."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("unexpected BeyondAIME revision")
    seen = set()
    rows = []
    for index, source_row in enumerate(parquet.read_table(source).to_pylist()):
        problem, answer = source_row["problem"], source_row["answer"]
        normalized = " ".join(problem.split()).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        if type(answer) is int:
            rows.append({"id": f"beyond-aime-{index:05d}", "problem": problem, "answer": answer})
    return sorted(rows, key=lambda row: order(row["id"]))[:48]


def comparison_rows(manifest: Path, summary: Path) -> list[dict]:
    """Freeze 15 pass and 15 fail cases from one observed pass@1 per problem."""
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        manifest_rows = list(csv.DictReader(file))
    result = json.loads(summary.read_text(encoding="utf-8"))
    outcomes = {}
    for row in result["rows"]:
        if row.get("checkpoints"):
            outcomes[row["id"]] = row["checkpoints"][0]["evaluation"]
    chosen = []
    for outcome in ("correct", "incorrect"):
        group = [row for row in manifest_rows if outcomes.get(row["id"]) == outcome]
        if len(group) < 15:
            raise ValueError(f"15 independently scored {outcome} cases are required, found {len(group)}")
        chosen.append(group[:15])
    return sorted(chosen[0] + chosen[1], key=lambda row: order(row["id"]))


def main() -> None:
    """Create one new frozen manifest and print its digest without printing answers."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("source", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("manifest", type=Path)
    compare.add_argument("summary", type=Path)
    for command in (screen, compare):
        command.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = (
        screen_rows(args.source) if args.stage == "screen"
        else comparison_rows(args.manifest, args.summary)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("id", "problem", "answer"))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows, sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__ == "__main__":
    main()
