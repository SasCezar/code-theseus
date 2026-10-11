from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from reverse_se.core.types import Repository

load_dotenv()

ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))

_WS_RE = re.compile(r"\s+")


def is_github_url(url: str) -> bool:
    return "github.com" in url.lower()


def normalize_url(url: str) -> str:
    """
    Normalize for dedup:
      - drop query + fragment
      - strip trailing slash
    """
    p = urlparse(url.strip())
    clean = f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
    return clean


def clean_text(s: str | None) -> str | None:
    if s is None:
        return None
    s = _WS_RE.sub(" ", s).strip()
    return s or None


def dedup_by_url_keep_first(repos: list[Repository]) -> list[Repository]:
    seen: dict[str, Repository] = {}
    for r in repos:
        if r.url and r.url not in seen:
            seen[r.url] = r
    return list(seen.values())


def read_text_utf8(file: Path) -> str:
    return file.read_text(encoding="utf-8", errors="ignore")
