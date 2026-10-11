#!/usr/bin/env python3
"""
collect_repo.py — Collect architecture + context data from a single repository.

Reads Arcan analysis output (component-*.csv files) and scans context files
(README, Dockerfiles, build configs, dependency manifests) to produce a single
JSONL record compatible with the reverse-se dataset format.

No external dependencies required — uses Python standard library only.
Run with:  python3 collect_repo.py [args]
       or: uv run collect_repo.py [args]   (if uv is installed)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# ── Language normalization ─────────────────────────────────────────────────────

_LANG_ALIASES: dict[str, str] = {
    "C": "c",
    "CPP": "cpp",
    "C++": "cpp",
    "JAVA": "java",
    "PYTHON": "python",
    "CSHARP": "csharp",
    "C#": "csharp",
}
_VALID_LANGS = "C, CPP, JAVA, PYTHON, CSHARP"


def _norm_lang(lang: str) -> str:
    key = lang.strip()
    for candidate in (key, key.upper()):
        if candidate in _LANG_ALIASES:
            return _LANG_ALIASES[candidate]
    raise ValueError(f"Unsupported language '{lang}'. Valid options: {_VALID_LANGS}")


# ── Context file constants (from context_file_extractor.py) ───────────────────

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
    "CMakeLists.txt",
    "meson.build",
    "Makefile",
    "makefile",
    "configure.ac",
    "configure.in",
    "*.mk",
    "*.cmake",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradlew",
    "gradlew.bat",
    "mvnw",
    "mvnw.cmd",
    "*.sln",
    "*.csproj",
    "Directory.Build.props",
    "Directory.Build.targets",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
]

ALL_CATEGORIES = ["docker", "dependencies", "build", "readme"]
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

LIKELY_DIRS = {".", "src", "app", "services", "service", "packages", "modules", "cmd"}

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


# ── Context file scanning helpers ─────────────────────────────────────────────


def _matches(rel_path_posix: str, patterns: list[str]) -> bool:
    base = Path(rel_path_posix).name
    return any(fnmatch(rel_path_posix, pat) or fnmatch(base, pat) for pat in patterns)


def _rel_depth(rel_posix: str) -> int:
    if rel_posix in ("", "."):
        return 0
    return rel_posix.count("/") + 1


def _phase_roots(repo_dir: Path) -> list[Path]:
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
    seen: set[Path] = set()
    uniq: list[Path] = []
    for r in roots:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def _bounded_walk(repo_dir: Path, roots: list[Path], max_depth: int, max_files: int):
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
            if _rel_depth(rel_root) > max_depth:
                dirs[:] = []
                continue
            for f in files:
                seen_files += 1
                if seen_files > max_files:
                    return
                yield root_p / f


def _dir_has_evidence(repo_dir: Path, d: Path, language: str) -> int:
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


def _read_file_full(path: Path, max_bytes: int) -> str | None:
    try:
        st = path.stat()
        if st.st_size > max_bytes:
            return None
        data = path.read_bytes()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _choose_preferred_dependency(
    repo_dir: Path, language: str, max_depth: int, max_files: int
) -> str | None:
    prefs = DEPENDENCY_PREFERENCES.get(language, [])
    if not prefs:
        return None
    roots = _phase_roots(repo_dir)
    candidates: list[str] = []
    for p in _bounded_walk(
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
            evidence = _dir_has_evidence(repo_dir, (repo_dir / rel).parent, language)
            return (depth, -evidence, rel)

        return sorted(hits, key=key)[0]
    return None


def _classify_other(rel: str) -> str | None:
    checks = {
        "docker": DOCKER_PATTERNS,
        "build": BUILD_PATTERNS,
        "readme": README_PATTERNS,
    }
    for cat in CATEGORY_PRECEDENCE:
        if cat == "dependencies":
            continue
        if _matches(rel, checks.get(cat, [])):
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
) -> list[dict[str, Any]]:
    """Scan repo for context files. Returns list of document dicts."""
    documents: list[dict] = []
    seen: set[str] = set()
    cat_counts: dict[str, int] = dict.fromkeys(CATEGORY_PRECEDENCE, 0)

    dep = _choose_preferred_dependency(
        repo_dir, language, max_depth=phaseA_depth, max_files=phaseA_max_files
    )
    if dep is None:
        roots = [repo_dir]
        prefs = DEPENDENCY_PREFERENCES.get(language, [])
        candidates: list[str] = []
        for p in _bounded_walk(
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
        content = _read_file_full(repo_dir / dep, max_bytes=max_bytes)
        if content is not None:
            documents.append(
                {"path": dep, "category": "dependencies", "content": content}
            )
            seen.add(dep)
            cat_counts["dependencies"] += 1

    def add_doc(rel: str, cat: str) -> None:
        if rel in seen:
            return
        if cat_counts.get(cat, 0) >= per_category_cap.get(cat, 10**9):
            return
        content = _read_file_full(repo_dir / rel, max_bytes=max_bytes)
        if content is None:
            return
        documents.append({"path": rel, "category": cat, "content": content})
        seen.add(rel)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    rootsA = _phase_roots(repo_dir)
    for p in _bounded_walk(
        repo_dir, roots=rootsA, max_depth=phaseA_depth, max_files=phaseA_max_files
    ):
        rel = p.relative_to(repo_dir).as_posix()
        if rel in seen:
            continue
        cat = _classify_other(rel)
        if not cat:
            continue
        if cat == "build" and _matches(rel, DEPENDENCY_PREFERENCES.get(language, [])):
            continue
        add_doc(rel, cat)

    if (
        cat_counts["readme"] == 0
        and cat_counts["docker"] == 0
        and cat_counts["build"] == 0
    ):
        for p in _bounded_walk(
            repo_dir,
            roots=[repo_dir],
            max_depth=phaseB_depth,
            max_files=phaseB_max_files,
        ):
            rel = p.relative_to(repo_dir).as_posix()
            if rel in seen:
                continue
            cat = _classify_other(rel)
            if not cat:
                continue
            if cat == "build" and _matches(
                rel, DEPENDENCY_PREFERENCES.get(language, [])
            ):
                continue
            add_doc(rel, cat)

    rank = {c: i for i, c in enumerate(CATEGORY_PRECEDENCE)}
    documents.sort(key=lambda d: (rank.get(d["category"], 999), d["path"]))
    return documents


# ── Arcan CSV reading (replaces pandas-based ArcanCSV) ────────────────────────

_CLEAN_EDGES_COLS = {
    "project",
    "versionId",
    "versionIndex",
    "versionDate",
    "InferSourceList",
    "Inferred",
    "LocationList",
    "SmellExtentByWeight",
    "edgeLabel",
    "from",
    "to",
    "edgeId",
}
_CLEAN_NODES_COLS = {
    "project",
    "versionId",
    "versionIndex",
    "versionDate",
    "AbstractnessMetric",
    "ChangeHasOccurred",
    "InstabilityMetric",
    "NumberOfChildren",
    "TotalAmountOfChanges",
    "filePathReal",
    "simpleName",
    "vertexLabel",
    "constructType",
    "IsNested",
    "filePathRelative",
}
_NODE_RENAME = {
    "ComponentType": "component_type",
    "FanIn": "fan_in",
    "FanOut": "fan_out",
    "LinesOfCode": "lines_of_code",
    "PageRank": "pagerank",
    "name": "name",
    "vertexId": "vertex_id",
}
_EDGE_RENAME = {
    "fromId": "source",
    "toId": "target",
    "Weight": "weight",
}


def _try_num(v: Any) -> Any:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        f = float(v)
        return None if f != f else f  # NaN → None
    except (ValueError, TypeError):
        return v


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def read_arcan_graph(
    graph_path: Path,
    package_only: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Read Arcan CSVs; return (nodes, edges, package_files)."""
    nodes = _read_csv(graph_path / "component-metrics.csv")
    edges = _read_csv(graph_path / "component-dependencies.csv")
    membership = _read_csv(graph_path / "component-membership.csv")

    # Build package → files mapping from membership
    pkg_files: dict[str, list[str]] = {}
    for row in membership:
        pkg = row.get("from", "")
        full = row.get("to", "")
        prefix = pkg + "."
        file_name = full[len(prefix) :] if full.startswith(prefix) else full
        pkg_files.setdefault(pkg, []).append(file_name)
    package_files = [{"package": p, "files": fs} for p, fs in pkg_files.items()]

    if package_only:
        nodes = [n for n in nodes if n.get("vertexLabel") == "container"]
        container_ids = {n["vertexId"] for n in nodes}
        edges = [
            e
            for e in edges
            if e.get("fromId") in container_ids and e.get("toId") in container_ids
        ]

    # Drop Arcan-internal columns
    edges = [{k: v for k, v in e.items() if k not in _CLEAN_EDGES_COLS} for e in edges]
    nodes = [{k: v for k, v in n.items() if k not in _CLEAN_NODES_COLS} for n in nodes]

    return nodes, edges, package_files


def normalize_nodes(nodes: list[dict]) -> list[dict]:
    return [
        {_NODE_RENAME[k]: _try_num(v) for k, v in n.items() if k in _NODE_RENAME}
        for n in nodes
    ]


def normalize_edges(edges: list[dict]) -> list[dict]:
    return [
        {_EDGE_RENAME[k]: _try_num(v) for k, v in e.items() if k in _EDGE_RENAME}
        for e in edges
    ]


def get_graph_hash(graph_path: Path) -> str | None:
    """Extract commit hash from the graphml filename (<project>_<hash>.graphml)."""
    gml = next(graph_path.glob("*.graphml"), None)
    if gml is None:
        return None
    parts = gml.stem.split("_", 1)
    return parts[1] if len(parts) > 1 else gml.stem


def safe_ratio(raw: int | float, compressed: int | float) -> float:
    if compressed <= 0:
        return float("inf")
    return float(raw) / float(compressed)


def build_graph(
    project: str,
    graph_path: Path,
    package_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build graph dict and stats from Arcan CSV files."""
    nodes, edges, package_files = read_arcan_graph(
        graph_path, package_only=package_only
    )
    norm_nodes = normalize_nodes(nodes)
    norm_edges = normalize_edges(edges)
    stats = {
        "num_nodes": len(norm_nodes),
        "num_edges": len(norm_edges),
        "num_packages": len(package_files),
    }
    graph = {
        "nodes": norm_nodes,
        "edges": norm_edges,
        "package_files": package_files,
    }
    return graph, stats


# ── Main collection logic ──────────────────────────────────────────────────────


def collect_repo(
    repo_path: Path,
    arcan_output: Path,
    language: str,
    output_file: Path,
    *,
    exclude_categories: set[str],
    max_bytes: int,
    phaseA_depth: int,
    phaseA_max_files: int,
    phaseB_depth: int,
    phaseB_max_files: int,
) -> None:
    project = repo_path.name

    per_category_cap = {
        "dependencies": 5,
        "docker": 5,
        "build": 5,
        "readme": 5,
    }
    # Zero-out excluded categories so they are never collected
    for cat in exclude_categories:
        per_category_cap[cat] = 0

    # Graph hash
    graph_hash = get_graph_hash(arcan_output)
    if graph_hash is None:
        raise SystemExit(
            f"No .graphml file found in {arcan_output}. "
            "Make sure Arcan ran successfully and the output directory is correct."
        )

    logging.info("Building package-level graph...")
    package_graph, package_stats = build_graph(project, arcan_output, package_only=True)
    package_files = package_graph.pop("package_files", [])

    logging.info("Building full component graph...")
    original_graph, original_stats = build_graph(
        project, arcan_output, package_only=False
    )
    original_graph.pop("package_files", None)

    logging.info("Scanning context files...")
    documents = scan_repo(
        repo_path,
        language,
        max_bytes=max_bytes,
        phaseA_depth=phaseA_depth,
        phaseA_max_files=phaseA_max_files,
        phaseB_depth=phaseB_depth,
        phaseB_max_files=phaseB_max_files,
        per_category_cap=per_category_cap,
    )

    if exclude_categories:
        documents = [
            d for d in documents if d.get("category") not in exclude_categories
        ]

    context_files = [
        {
            "path": d.get("path"),
            "category": d.get("category"),
            "content": d.get("content"),
        }
        for d in documents
    ]

    result = {
        "repo_id": project,
        "name": project,
        "info": {
            "url": None,
            "description": None,
            "language": language,
            "archived": None,
            "pushed_at": None,
            "stargazers_count": None,
            "hash": graph_hash,
        },
        "package_stats": package_stats,
        "original_stats": original_stats,
        "package_only_graph": package_graph,
        "original_graph": original_graph,
        "package_files": package_files,
        "context_files": context_files,
        "node_compression_ratio": safe_ratio(
            original_stats["num_nodes"], package_stats["num_nodes"]
        ),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")

    logging.info(
        "Done. Nodes: %d (pkg) / %d (full) | Context files: %d | Output: %s",
        package_stats["num_nodes"],
        original_stats["num_nodes"],
        len(context_files),
        output_file,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Collect architecture + context data from a single repository.\n"
            "Produces a JSONL record compatible with the reverse-se dataset format.\n\n"
            "No external dependencies — uses Python standard library only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--repo-path",
        required=True,
        help="Path to the repository source code.",
    )
    ap.add_argument(
        "--arcan-output",
        required=True,
        help=(
            "Path to Arcan's output directory for this repo "
            "(the folder containing component-metrics.csv, component-dependencies.csv, etc.)."
        ),
    )
    ap.add_argument(
        "--language",
        required=True,
        help=f"Programming language of the repository. One of: {_VALID_LANGS}",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path.",
    )
    ap.add_argument(
        "--exclude-categories",
        default="",
        help=(
            "Comma-separated list of context file categories to exclude.\n"
            "Choices: readme, docker, build, dependencies\n"
            "Example: --exclude-categories readme,docker"
        ),
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=300_000,
        help="Skip context files larger than this many bytes (default: 300000).",
    )
    ap.add_argument("--phaseA-depth", type=int, default=2)
    ap.add_argument("--phaseA-max-files", type=int, default=20_000)
    ap.add_argument("--phaseB-depth", type=int, default=4)
    ap.add_argument("--phaseB-max-files", type=int, default=60_000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        language = _norm_lang(args.language)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    exclude_categories: set[str] = set()
    if args.exclude_categories:
        for cat in args.exclude_categories.split(","):
            cat = cat.strip().lower()
            if cat and cat not in ALL_CATEGORIES:
                raise SystemExit(
                    f"Unknown category '{cat}'. Valid: {', '.join(ALL_CATEGORIES)}"
                )
            if cat:
                exclude_categories.add(cat)

    repo_path = Path(args.repo_path).expanduser().resolve()
    arcan_output = Path(args.arcan_output).expanduser().resolve()
    output_file = Path(args.output).expanduser().resolve()

    for p, label in [(repo_path, "--repo-path"), (arcan_output, "--arcan-output")]:
        if not p.exists():
            raise SystemExit(f"{label}: path does not exist: {p}")

    collect_repo(
        repo_path=repo_path,
        arcan_output=arcan_output,
        language=language,
        output_file=output_file,
        exclude_categories=exclude_categories,
        max_bytes=args.max_bytes,
        phaseA_depth=args.phaseA_depth,
        phaseA_max_files=args.phaseA_max_files,
        phaseB_depth=args.phaseB_depth,
        phaseB_max_files=args.phaseB_max_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
