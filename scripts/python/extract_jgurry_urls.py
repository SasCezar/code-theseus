from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from utils import (
    ROOT_DIR,
    clean_text,
    dedup_by_url_keep_first,
    is_github_url,
    normalize_url,
    read_text_utf8,
)

from reverse_se.core.types import Repository


def none_if_nan(v: Any):
    """
    Convert pandas/NumPy NaN or NA values to None so Pydantic accepts them.
    """
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN check
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def parse_pythonframeworks_html(
    file: Path, language: str = "python"
) -> tuple[list[Repository], pd.DataFrame]:
    """
    Returns:
      - github_repos: list[Repository] for GitHub URLs (includes language)
      - web_df: DataFrame for non-GitHub URLs with columns:
                name, url, awesome_descr, language
    """
    html = read_text_utf8(file)
    soup = BeautifulSoup(html, "html.parser")

    cards = soup.select("div.card")

    rows: list[dict[str, str | None]] = []
    for card in cards:
        a = card.select_one("a[href]")
        if not a:
            continue

        url = (a.get("href") or "").strip()
        if not url:
            continue

        img = card.select_one("img[alt]")
        name = clean_text(img.get("alt") if img else None)

        if not name:
            title = card.select_one(".card-title")
            name = clean_text(title.get_text(" ", strip=True) if title else None)

        if not name:
            continue

        descr_el = card.select_one(".card-text")
        descr = clean_text(descr_el.get_text(" ", strip=True) if descr_el else None)

        rows.append(
            {
                "name": name,
                "url": url,
                "awesome_descr": descr,
                "language": language,
            }
        )

    df = pd.DataFrame(rows, columns=["name", "url", "awesome_descr", "language"])

    # Normalize + dedup by URL
    if not df.empty:
        df["url"] = df["url"].apply(normalize_url)
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    github_df = df[df["url"].apply(is_github_url)].copy()
    web_df = df[~df["url"].apply(is_github_url)].copy()

    github_repos = [
        Repository(
            name=str(row["name"]),
            url=none_if_nan(row["url"]),
            description=none_if_nan(row["awesome_descr"]),
            language=none_if_nan(row["language"]),
        )
        for _, row in github_df.iterrows()
    ]

    return github_repos, web_df


def extract_pythonframeworks_html_urls(
    html_file: Path,
    language: str = "python",
    out_dir: Path | None = None,
) -> tuple[list[Repository], pd.DataFrame]:
    """
    Matches the Markdown extractor behavior:
      - writes repos_github.csv and repos_webpage.csv
      - returns (github_repos_all, web_df_all)
    """
    base_path = out_dir or (ROOT_DIR / "data/raw")

    gh_repos, web_df = parse_pythonframeworks_html(html_file, language=language)

    gh_repos = dedup_by_url_keep_first(gh_repos)

    github_df = pd.DataFrame(
        {
            "name": [r.name for r in gh_repos],
            "url": [r.url for r in gh_repos],
            "awesome_descr": [r.description for r in gh_repos],
            "language": [r.language for r in gh_repos],
        },
        columns=["name", "url", "awesome_descr", "language"],
    ).drop_duplicates(subset=["url"])

    web_df_all = (
        web_df
        if not web_df.empty
        else pd.DataFrame(columns=["name", "url", "awesome_descr", "language"])
    ).drop_duplicates(subset=["url"])

    (base_path / "jgujerry_repos_github.csv").write_text(
        github_df.to_csv(index=False), encoding="utf-8"
    )
    (base_path / "jgujerry_repos_webpage.csv").write_text(
        web_df_all.to_csv(index=False), encoding="utf-8"
    )

    return gh_repos, web_df_all


if __name__ == "__main__":
    html_path = ROOT_DIR / "data/raw/raw-list-python/jgujerry-python-frameworks.html"
    extract_pythonframeworks_html_urls(html_path)
