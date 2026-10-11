#!/usr/bin/env python3
"""Merge the per-source repo-URL CSVs (extract_jgurry_urls.py,
extract_awesome_urls.py, gitranking2csv.py) and dedupe by url.

Usage:
    uv run scripts/python/deduplicate.py
    uv run scripts/python/deduplicate.py --inputs a.csv b.csv --output out.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from utils import ROOT_DIR

DEFAULT_INPUTS = [
    ROOT_DIR / "data/raw/jgujerry_repos_github.csv",
    ROOT_DIR / "data/raw/repo-lists/repos_github.csv",
    ROOT_DIR / "data/raw/github_projects.csv",
]
DEFAULT_OUTPUT = ROOT_DIR / "data/raw/full_list_deduped.csv"


def deduplicate(inputs: list[Path], output: Path) -> None:
    frames = []
    for path in inputs:
        if not path.exists():
            print(f"  [skip] not found: {path}")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No input CSVs found.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} deduped rows -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Source CSVs to merge (each must have a 'url' column)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    deduplicate(args.inputs, args.output)


if __name__ == "__main__":
    main()
