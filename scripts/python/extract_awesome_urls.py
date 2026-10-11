from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from utils import (
    ROOT_DIR,
    dedup_by_url_keep_first,
    is_github_url,
)

from reverse_se.core.types import Repository

MD_BULLET_RE = re.compile(
    r"""^\s*-\s+\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)\s*-\s*(?P<descr>.+?)\s*$""",
    re.MULTILINE,
)


def parse_doc(file: Path, language: str) -> tuple[list[Repository], pd.DataFrame]:
    """
    Returns:
      - github_repos: list[Repository] for GitHub URLs (includes language)
      - web_df: DataFrame for non-GitHub URLs with columns: name, url, awesome_descr, language
    """
    content = file.read_text(encoding="utf-8", errors="ignore")

    items: list[dict[str, str]] = []
    for m in MD_BULLET_RE.finditer(content):
        items.append(
            {
                "name": m.group("name").strip(),
                "url": m.group("url").strip(),
                "awesome_descr": m.group("descr").strip(),
                "language": language,
            }
        )

    github_repos: list[Repository] = []
    web_rows: list[dict[str, str]] = []

    for it in items:
        if is_github_url(it["url"]):
            github_repos.append(
                Repository(
                    name=it["name"],
                    url=it["url"],
                    description=it["awesome_descr"],
                    language=it["language"],
                )
            )
        else:
            web_rows.append(it)

    web_df = pd.DataFrame(
        web_rows, columns=["name", "url", "awesome_descr", "language"]
    )
    return github_repos, web_df


def iter_md_files(folder: Path) -> Iterable[Path]:
    yield from folder.glob("*.md")


def extract_awesome_urls():
    base_path: Path = ROOT_DIR / "data/raw/repo-lists"
    folders = [
        base_path / "raw-list-c",
        base_path / "raw-list-cpp",
        base_path / "raw-list-java",
        base_path / "raw-list-python",
        base_path / "raw-list-csharp",
    ]

    github_repos_all: list[Repository] = []
    web_dfs: list[pd.DataFrame] = []

    for folder in folders:
        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        language = folder.name.split("-", 2)[-1]  # "raw-list-python" -> "python"
        print(f"Extracting entries from {language} repositories...")

        for file in iter_md_files(folder):
            gh_repos, web_df = parse_doc(file, language=language)
            github_repos_all.extend(gh_repos)
            if not web_df.empty:
                web_dfs.append(web_df)

    # --- GitHub: deduplicate
    github_repos_all = dedup_by_url_keep_first(github_repos_all)

    github_df = pd.DataFrame(
        {
            "name": [r.name for r in github_repos_all],
            "url": [r.url for r in github_repos_all],
            "awesome_descr": [r.description for r in github_repos_all],
            "language": [r.language for r in github_repos_all],
        },
        columns=["name", "url", "awesome_descr", "language"],
    ).drop_duplicates(subset=["url"])

    web_df_all = (
        pd.concat(web_dfs, ignore_index=True)
        if web_dfs
        else pd.DataFrame(columns=["name", "url", "awesome_descr", "language"])
    ).drop_duplicates(subset=["url"])

    (base_path / "repos_github.csv").write_text(
        github_df.to_csv(index=False), encoding="utf-8"
    )
    (base_path / "repos_webpage.csv").write_text(
        web_df_all.to_csv(index=False), encoding="utf-8"
    )

    return github_repos_all, web_df_all


if __name__ == "__main__":
    extract_awesome_urls()
