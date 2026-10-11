#!/usr/bin/env python3
"""Fetch GitHub repo metadata (stars, language, pushed_at, archived, ...) for
every url in a repo-list CSV, via the GitHub API.

Input default chains from deduplicate.py's default output; output feeds
filter_dataset.py's default input.

Usage:
    uv run scripts/python/fetch_github_info.py
    uv run scripts/python/fetch_github_info.py --input other.csv --output other.jsonl
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from utils import ROOT_DIR

from reverse_se.fetcher.github import GitHubRepoBatchFetcher

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

DEFAULT_INPUT = ROOT_DIR / "data/raw/full_list_deduped.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data/raw/full_list_info.jsonl"


def fetch_github_repos(repo_list_path: Path, out_path: Path) -> None:
    df = pd.read_csv(repo_list_path)
    repo_urls = df["url"].tolist()

    batch_fetcher = GitHubRepoBatchFetcher(GITHUB_TOKEN, batch_size=50)

    repos, stats = batch_fetcher.fetch_projects(
        repo_urls,
        show_progress=True,
        return_stats=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in repos:
            f.write(r.model_dump_json() + "\n")

    print("\n=== Fetch summary ===")
    print(f"Total URLs:         {stats.total_urls}")
    print(f"Valid owner/repo:   {stats.total_valid_repos}")
    print(f"Success:            {stats.success}")
    print(f"Failures total:     {sum(stats.failures.values())}")
    print("\nFailures by type:")
    for k, v in stats.failures.most_common():
        print(f"  - {k}: {v}")

    print("\nExamples (up to 5 each):")
    for k, examples in stats.failure_examples.items():
        if not examples:
            continue
        print(f"\n{k}:")
        for ex in examples:
            print(f"  {ex}")

    print(f"\nWrote {len(repos)} records -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Repo-list CSV with a 'url' column",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fetch_github_repos(args.input, args.output)


if __name__ == "__main__":
    main()
