#!/usr/bin/env python3
"""Filter fetched GitHub metadata down to a language TSV (url, language) for
repos that are active, non-archived, and above a star threshold.

Input default chains from fetch_github_info.py's default output; output
(the filter_*.tsv) feeds prefetch_repo.sh and run_arcan_docker.sh.

Usage:
    uv run scripts/python/filter_dataset.py
    uv run scripts/python/filter_dataset.py --min-year 2022 --min-stars 100
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from utils import ROOT_DIR

mapping = {
    "python": "PYTHON",
    "Python": "PYTHON",
    "Java": "JAVA",
    "java": "JAVA",
    "c": "C",
    "c++": "CPP",
    "C++": "CPP",
    "C#": "CSHARP",
    "c#": "CSHARP",
}

ALLOWED_LANGUAGES = {"PYTHON", "JAVA", "C", "CPP", "CSHARP"}

DEFAULT_INPUT = ROOT_DIR / "data/raw/full_list_info.jsonl"
DEFAULT_INFO_CSV = ROOT_DIR / "data/raw/full_list_info.csv"


def language_mapping(lang):
    return mapping.get(lang, lang)


def filter_dataset(
    input_jsonl: Path,
    info_csv: Path,
    output_tsv: Path,
    min_year: int,
    min_stars: int,
) -> None:
    with open(input_jsonl) as inf:
        lines = [json.loads(line) for line in inf]

    df = pd.DataFrame(lines)
    df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    df["language"] = df["language"].map(language_mapping)
    df = df[df["language"].isin(ALLOWED_LANGUAGES)]

    info_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(info_csv, index=False)

    df["pushed_at"] = pd.to_datetime(df["pushed_at"], errors="coerce", utc=True)
    cutoff_date = pd.Timestamp(f"{min_year}-01-01").tz_localize("UTC")

    mask = (
        df["archived"].fillna(False).eq(False)
        & pd.to_numeric(df["stargazers_count"], errors="coerce").fillna(0).ge(min_stars)
        & df["pushed_at"].ge(cutoff_date)
    )
    df_filtered = df.loc[mask, ["url", "language"]].copy()

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_tsv, index=False, header=None, sep="\t")
    print(f"Info CSV:   {info_csv} ({len(df)} rows)")
    print(f"Filter TSV: {output_tsv} ({len(df_filtered)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--info-csv", type=Path, default=DEFAULT_INFO_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: data/raw/filter_{min-stars}_{min-year}_list.tsv",
    )
    parser.add_argument("--min-year", type=int, default=2021)
    parser.add_argument("--min-stars", type=int, default=50)
    args = parser.parse_args()

    output = args.output or ROOT_DIR / (
        f"data/raw/filter_{args.min_stars}_{args.min_year}_list.tsv"
    )
    filter_dataset(args.input, args.info_csv, output, args.min_year, args.min_stars)


if __name__ == "__main__":
    main()
