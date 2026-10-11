#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass

from tqdm import tqdm


def parse_repo_dirname(dirname: str) -> tuple[str, str] | None:
    if "|" not in dirname:
        return None
    user, repo = dirname.split("|", 1)
    user = user.strip()
    repo = repo.strip()
    if not user or not repo:
        return None
    return user, repo


@dataclass(frozen=True)
class Repo:
    entry: str  # dir name: user|repo
    repo_dir: str  # absolute path
    repo_url: str  # github url


def build_repos(input_dir: str) -> list[Repo]:
    repos: list[Repo] = []
    with os.scandir(input_dir) as it:
        for dent in it:
            if not dent.is_dir(follow_symlinks=False):
                continue
            parsed = parse_repo_dirname(dent.name)
            if not parsed:
                print(
                    f"WARN: skipping directory not matching user|repo: {dent.name}",
                    file=sys.stderr,
                )
                continue
            user, repo = parsed
            repos.append(
                Repo(
                    entry=dent.name,
                    repo_dir=os.path.abspath(dent.path),
                    repo_url=f"https://github.com/{user}/{repo}",
                )
            )

    repos.sort(key=lambda r: r.entry)
    return repos


def run_tokei_repo(repo_dir: str) -> dict | None:
    """
    Run tokei on a single repo directory (NO --files).
    Returns parsed JSON dict on success, None on failure.
    """
    try:
        proc = subprocess.run(
            ["tokei", "--output", "json", repo_dir],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("ERROR: tokei not found in PATH", file=sys.stderr)
        return None

    if proc.returncode != 0:
        msg = proc.stderr.strip() or f"tokei exited with code {proc.returncode}"
        print(f"ERROR: tokei failed for {repo_dir}: {msg}", file=sys.stderr)
        return None

    out = proc.stdout.strip()
    if not out:
        return None

    try:
        return json.loads(out)
    except json.JSONDecodeError:
        print(f"ERROR: invalid JSON from tokei for {repo_dir}", file=sys.stderr)
        return None


def top_language_from_tokei_json(data: dict) -> str | None:
    """
    Given tokei JSON output for one directory, return the language
    with the highest 'code' count, or None if no language stats.
    """
    best_lang: str | None = None
    best_code = -1

    for lang, stats in data.items():
        if lang == "Total" or not isinstance(stats, dict):
            continue
        code = stats.get("code")
        if not isinstance(code, int) or code <= 0:
            continue
        if code > best_code:
            best_code = code
            best_lang = lang

    return best_lang


def main() -> int:
    input_dir = os.environ.get("INPUT_DIR", "/repos")
    output_tsv = os.environ.get("OUTPUT_TSV", "/out/results.tsv")

    if not os.path.isdir(input_dir):
        print(f"ERROR: INPUT_DIR is not a directory: {input_dir}", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(output_tsv) or ".", exist_ok=True)

    repos = build_repos(input_dir)
    if not repos:
        open(output_tsv, "w", encoding="utf-8").close()
        print(f"Wrote 0 rows to {output_tsv}", file=sys.stderr)
        return 0

    wrote = 0
    with open(output_tsv, "w", encoding="utf-8") as f:
        for r in tqdm(
            repos,
            desc="Analyzing repos",
            unit="repo",
            disable=not sys.stderr.isatty(),
            file=sys.stderr,
        ):
            data = run_tokei_repo(r.repo_dir)
            if not data:
                continue

            top_lang = top_language_from_tokei_json(data)
            if not top_lang:
                continue

            f.write(f"{r.repo_url}\t{top_lang}\n")
            wrote += 1

    print(
        f"Wrote {wrote} rows to {output_tsv} (scanned {len(repos)} repos, tokei per repo)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
