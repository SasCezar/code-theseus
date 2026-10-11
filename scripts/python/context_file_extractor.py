#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path

from tqdm import tqdm

TSV_LANGS = {"C", "CPP", "JAVA", "PYTHON", "CSHARP"}

LANG_ALIASES = {
    "C": "c",
    "CPP": "cpp",
    "C++": "cpp",
    "JAVA": "java",
    "PYTHON": "python",
    "CSHARP": "csharp",
    "C#": "csharp",
}

_GH_HTTP_RE = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+)(?:/.*)?$", re.IGNORECASE
)


def norm_lang_from_tsv(lang: str) -> str:
    key = (lang or "").strip()
    if not key:
        raise ValueError("Empty language")
    if key in LANG_ALIASES:
        return LANG_ALIASES[key]
    key_up = key.upper()
    if key_up in LANG_ALIASES:
        return LANG_ALIASES[key_up]
    raise ValueError(
        f"Unsupported language '{lang}'. Expected one of: {sorted(TSV_LANGS)}"
    )


def extract_owner_repo_from_url(s: str) -> tuple[str, str] | None:
    s = (s or "").strip()
    if not s:
        return None
    m = _GH_HTTP_RE.match(s)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def load_language_tsv(tsv_path: Path) -> dict[str, str]:
    """
    TSV format (2 columns):
      https://github.com/<name>/<repo>\t<LANG>

    Produces mapping:
      "<name>|<repo>" -> normalized_lang (c/cpp/java/python/csharp)
    """
    mapping: dict[str, str] = {}
    lines = tsv_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for i, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"Bad TSV line {i}: expected 2 columns separated by TAB")

        key_raw = parts[0].strip()
        lang_raw = parts[1].strip()
        lang = norm_lang_from_tsv(lang_raw)

        # Accept either github URL OR already "name|repo"
        owner_repo = extract_owner_repo_from_url(key_raw)
        if owner_repo:
            owner, repo = owner_repo
            dir_key = f"{owner}|{repo}"
            mapping[dir_key] = lang
            mapping[dir_key.lower()] = lang
            continue

        # Fallback: allow direct "name|repo" keys if user gives them
        if "|" in key_raw and "/" not in key_raw:
            mapping[key_raw] = lang
            mapping[key_raw.lower()] = lang
            continue

        raise ValueError(
            f"Bad TSV line {i}: first column must be a GitHub URL "
            f"(https://github.com/name/repo) or a 'name|repo' key. Got: {key_raw!r}"
        )

    return mapping


def lookup_repo_language(repo_dir: Path, lang_map: dict[str, str]) -> str | None:
    """
    Repo dir name is expected to be: <name>|<repo>.
    We simply look that up in the TSV-derived mapping.
    """
    key = repo_dir.name
    if key in lang_map:
        return lang_map[key]
    key_l = key.lower()
    if key_l in lang_map:
        return lang_map[key_l]
    return None


DEPENDENCY_PREFERENCES: dict[str, list[str]] = {
    "python": [
        "pyproject.toml",
        "poetry.lock",
        "Pipfile",
        "Pipfile.lock",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.cfg",
        "setup.py",
        "environment.yml",
        "conda.yml",
    ],
    "java": [
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradle.properties",
        "mvnw",
        "mvnw.cmd",
        "gradlew",
        "gradlew.bat",
    ],
    "c": [
        "CMakeLists.txt",
        "meson.build",
        "Makefile",
        "makefile",
        "configure.ac",
        "configure.in",
        "vcpkg.json",
        "conanfile.txt",
        "conanfile.py",
    ],
    "cpp": [
        "CMakeLists.txt",
        "meson.build",
        "Makefile",
        "makefile",
        "configure.ac",
        "configure.in",
        "vcpkg.json",
        "conanfile.txt",
        "conanfile.py",
        "*.vcxproj",
        "*.sln",
    ],
    "csharp": [
        "*.csproj",
        "*.sln",
        "Directory.Packages.props",
        "packages.config",
        "global.json",
        "nuget.config",
    ],
}

DOCKER_PATTERNS = [
    "Dockerfile",
    "dockerfile",
    "Dockerfile.*",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
]

README_PATTERNS = [
    "README",
    "README.*",
]

BUILD_PATTERNS = [
    # C/C++
    "CMakeLists.txt",
    "meson.build",
    "Makefile",
    "makefile",
    "configure.ac",
    "configure.in",
    "*.mk",
    "*.cmake",
    # Java
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradlew",
    "gradlew.bat",
    "mvnw",
    "mvnw.cmd",
    # .NET
    "*.sln",
    "*.csproj",
    "Directory.Build.props",
    "Directory.Build.targets",
    # CI build entrypoints
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
]

CATEGORY_PRECEDENCE = ["docker", "dependencies", "build", "readme"]

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "vendor",
    "third_party",
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    ".venv",
    "venv",
    ".tox",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".gradle",
    ".idea",
    ".vscode",
    ".cache",
    ".turbo",
    ".next",
    "coverage",
    "libs",
    ".github",
    ".gitlab",
    ".circleci",
}

LIKELY_DIRS = {
    ".",
    "src",
    "app",
    "services",
    "service",
    "packages",
    "modules",
    "cmd",
}

LANG_EVIDENCE: dict[str, list[str]] = {
    "java": ["src/main", "src/test", "pom.xml", "build.gradle", "build.gradle.kts"],
    "python": [
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "src",
        "__init__.py",
    ],
    "csharp": [
        "*.csproj",
        "*.sln",
        "Program.cs",
        "Directory.Build.props",
        "Directory.Packages.props",
    ],
    "c": ["CMakeLists.txt", "Makefile", "meson.build", "include", "src"],
    "cpp": [
        "CMakeLists.txt",
        "Makefile",
        "meson.build",
        "include",
        "src",
        "*.vcxproj",
        "*.sln",
    ],
}


def matches(rel_path_posix: str, patterns: list[str]) -> bool:
    base = Path(rel_path_posix).name
    return any(fnmatch(rel_path_posix, pat) or fnmatch(base, pat) for pat in patterns)


def rel_depth(rel_posix: str) -> int:
    if rel_posix in ("", "."):
        return 0
    return rel_posix.count("/") + 1


def phase_roots(repo_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for d in LIKELY_DIRS:
        roots.append(repo_dir / d)

    try:
        for child in repo_dir.iterdir():
            if (
                child.is_dir()
                and child.name not in SKIP_DIRS
                and not child.name.startswith(".")
            ):
                roots.append(child)
    except Exception:
        pass

    uniq: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def bounded_walk(
    repo_dir: Path, roots: list[Path], max_depth: int, max_files: int
) -> Iterable[Path]:
    seen_files = 0
    repo_dir = repo_dir.resolve()

    for start in roots:
        start = start.resolve()
        if not start.exists() or not start.is_dir():
            continue
        try:
            start.relative_to(repo_dir)
        except ValueError:
            continue

        for root, dirs, files in os.walk(start):
            root_p = Path(root)
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            rel_root = root_p.relative_to(repo_dir).as_posix()
            if rel_depth(rel_root) > max_depth:
                dirs[:] = []
                continue

            for f in files:
                seen_files += 1
                if seen_files > max_files:
                    return
                yield root_p / f


def dir_has_evidence(repo_dir: Path, d: Path, language: str) -> int:
    score = 0
    evidence = LANG_EVIDENCE.get(language, [])
    try:
        entries = list(d.iterdir())
    except Exception:
        return 0

    names = [e.name for e in entries]
    for ev in evidence:
        if "/" in ev:
            if (d / ev).exists():
                score += 3
        else:
            if any(fnmatch(n, ev) for n in names):
                score += 2
    return score


def read_file_full(path: Path, max_bytes: int) -> str | None:
    try:
        st = path.stat()
        if st.st_size > max_bytes:
            return None
        data = path.read_bytes()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def choose_preferred_dependency(
    repo_dir: Path, language: str, max_depth: int, max_files: int
) -> str | None:
    prefs = DEPENDENCY_PREFERENCES.get(language, [])
    if not prefs:
        return None

    roots = phase_roots(repo_dir)
    candidates: list[str] = []

    for p in bounded_walk(
        repo_dir, roots=roots, max_depth=max_depth, max_files=max_files
    ):
        rel = p.relative_to(repo_dir).as_posix()
        for pat in prefs:
            if fnmatch(rel, pat) or fnmatch(p.name, pat):
                candidates.append(rel)
                break

    if not candidates:
        return None

    candidates = sorted(set(candidates))

    for pat in prefs:
        hits = [c for c in candidates if fnmatch(c, pat) or fnmatch(Path(c).name, pat)]
        if not hits:
            continue

        def key(rel: str):
            depth = rel.count("/")
            evidence = dir_has_evidence(repo_dir, (repo_dir / rel).parent, language)
            return (depth, -evidence, rel)

        return sorted(hits, key=key)[0]

    return None


def classify_other(rel: str) -> str | None:
    checks = {
        "docker": DOCKER_PATTERNS,
        "build": BUILD_PATTERNS,
        "readme": README_PATTERNS,
    }
    for cat in CATEGORY_PRECEDENCE:
        if cat == "dependencies":
            continue
        if matches(rel, checks.get(cat, [])):
            return cat
    return None


def scan_repo(
    repo_dir: Path,
    language: str,
    *,
    max_bytes: int,
    phaseA_depth: int,
    phaseA_max_files: int,
    phaseB_depth: int,
    phaseB_max_files: int,
    per_category_cap: dict[str, int],
) -> dict:
    documents: list[dict] = []
    seen: set[str] = set()
    cat_counts = dict.fromkeys(CATEGORY_PRECEDENCE, 0)

    dep = choose_preferred_dependency(
        repo_dir, language, max_depth=phaseA_depth, max_files=phaseA_max_files
    )

    if dep is None:
        roots = [repo_dir]
        prefs = DEPENDENCY_PREFERENCES.get(language, [])
        candidates: list[str] = []
        for p in bounded_walk(
            repo_dir, roots=roots, max_depth=phaseB_depth, max_files=phaseB_max_files
        ):
            rel = p.relative_to(repo_dir).as_posix()
            for pat in prefs:
                if fnmatch(rel, pat) or fnmatch(p.name, pat):
                    candidates.append(rel)
                    break
        if candidates:
            candidates = sorted(set(candidates))
            for pat in prefs:
                hits = [
                    c
                    for c in candidates
                    if fnmatch(c, pat) or fnmatch(Path(c).name, pat)
                ]
                if hits:
                    dep = sorted(hits, key=lambda s: (s.count("/"), s))[0]
                    break

    if dep:
        content = read_file_full(repo_dir / dep, max_bytes=max_bytes)
        if content is not None:
            documents.append(
                {"path": dep, "category": "dependencies", "content": content}
            )
            seen.add(dep)
            cat_counts["dependencies"] += 1

    def add_doc(rel: str, cat: str):
        if rel in seen:
            return
        if cat_counts.get(cat, 0) >= per_category_cap.get(cat, 10**9):
            return
        content = read_file_full(repo_dir / rel, max_bytes=max_bytes)
        if content is None:
            return
        documents.append({"path": rel, "category": cat, "content": content})
        seen.add(rel)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    rootsA = phase_roots(repo_dir)
    for p in bounded_walk(
        repo_dir, roots=rootsA, max_depth=phaseA_depth, max_files=phaseA_max_files
    ):
        rel = p.relative_to(repo_dir).as_posix()
        if rel in seen:
            continue
        cat = classify_other(rel)
        if not cat:
            continue
        if cat == "build" and matches(rel, DEPENDENCY_PREFERENCES.get(language, [])):
            continue
        add_doc(rel, cat)

    if (
        cat_counts["readme"] == 0
        and cat_counts["docker"] == 0
        and cat_counts["build"] == 0
    ):
        for p in bounded_walk(
            repo_dir,
            roots=[repo_dir],
            max_depth=phaseB_depth,
            max_files=phaseB_max_files,
        ):
            rel = p.relative_to(repo_dir).as_posix()
            if rel in seen:
                continue
            cat = classify_other(rel)
            if not cat:
                continue
            if cat == "build" and matches(
                rel, DEPENDENCY_PREFERENCES.get(language, [])
            ):
                continue
            add_doc(rel, cat)

    rank = {c: i for i, c in enumerate(CATEGORY_PRECEDENCE)}
    documents.sort(key=lambda d: (rank.get(d["category"], 999), d["path"]))

    return {"repo": repo_dir.name, "language": language, "documents": documents}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "parent_dir", help="Directory containing repos as immediate subdirectories"
    )
    ap.add_argument(
        "--language-tsv",
        required=True,
        help="TSV mapping: https://github.com/<name>/<repo>\\t<LANG> (LANG ALL CAPS: C, CPP, JAVA, PYTHON, CSHARP)",
    )
    ap.add_argument("--out", required=True, help="Output JSONL file")
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=300_000,
        help="Skip files larger than this (no truncation).",
    )
    ap.add_argument("--phaseA-depth", type=int, default=2)
    ap.add_argument("--phaseA-max-files", type=int, default=20000)
    ap.add_argument("--phaseB-depth", type=int, default=3)
    ap.add_argument("--phaseB-max-files", type=int, default=60000)
    ap.add_argument("--include-hidden", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parent: Path = Path(args.parent_dir).expanduser().resolve()
    if not parent.is_dir():
        raise SystemExit(f"Not a directory: {parent}")

    tsv_path: Path = Path(args.language_tsv).expanduser().resolve()
    if not tsv_path.is_file():
        raise SystemExit(f"Not a file: {tsv_path}")

    lang_map: dict[str, str] = load_language_tsv(tsv_path)

    per_category_cap = {
        "dependencies": 5,
        "docker": 5,
        "build": 5,
        "readme": 1,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repos = [
        p
        for p in sorted(parent.iterdir(), key=lambda p: p.name.lower())
        if p.is_dir() and (args.include_hidden or not p.name.startswith("."))
    ]

    success = 0
    fail = 0

    with out_path.open("w", encoding="utf-8") as f:
        bar = tqdm(repos, desc="Scanning repos", unit="repo", dynamic_ncols=True)
        for repo in bar:
            # Update live counters in the bar
            bar.set_postfix(success=success, fail=fail)

            try:
                lang = lookup_repo_language(repo, lang_map)
                if not lang:
                    fail += 1
                    logging.error(
                        "%s: language not found in TSV (expected key like 'name|repo')",
                        repo.name,
                    )
                    continue

                rec = scan_repo(
                    repo,
                    lang,
                    max_bytes=args.max_bytes,
                    phaseA_depth=args.phaseA_depth,
                    phaseA_max_files=args.phaseA_max_files,
                    phaseB_depth=args.phaseB_depth,
                    phaseB_max_files=args.phaseB_max_files,
                    per_category_cap=per_category_cap,
                )

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                success += 1

            except Exception as e:
                fail += 1
                logging.exception("%s: %s", repo.name, e)
                continue

        # final update
        bar.set_postfix(success=success, fail=fail)

    logging.info("Done. Success=%d Fail=%d Output=%s", success, fail, str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
