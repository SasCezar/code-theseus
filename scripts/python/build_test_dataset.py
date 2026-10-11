from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

# Reuse Arcan + normalisation helpers from combine_arcan_w_context
from combine_arcan_w_context import (
    build_graph,
    get_graph_hash,
    normalize_context_files,
    safe_ratio,
)

# Reuse scanning logic from context_file_extractor
from context_file_extractor import (
    norm_lang_from_tsv,
    scan_repo,
)
from tqdm import tqdm


def load_language_tsv(tsv_path: Path) -> dict[str, str]:
    """Load a TSV mapping repo folder names to languages.

    Accepted first-column formats (all map to the folder name as key):
      - Plain folder name:          my_project
      - owner|repo pair:            acme|my_project
      - Full GitHub URL:            https://github.com/acme/my_project

    Second column is the language (case-insensitive: C, CPP, JAVA, PYTHON, CSHARP).
    """
    import re

    _GH_RE = re.compile(
        r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$",
        re.IGNORECASE,
    )

    mapping: dict[str, str] = {}
    lines = tsv_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(
                f"Bad TSV line {i}: expected 2 tab-separated columns, got: {line!r}"
            )

        key_raw = parts[0].strip()
        lang_raw = parts[1].strip()
        lang = norm_lang_from_tsv(lang_raw)

        # GitHub URL → use just the repo name as key
        m = _GH_RE.match(key_raw)
        if m:
            repo_name = m.group(2)
            mapping[repo_name] = lang
            mapping[repo_name.lower()] = lang
            continue

        # owner|repo → use the repo part (after the |) as key, and the full pair
        if "|" in key_raw:
            _, repo_name = key_raw.rsplit("|", 1)
            mapping[repo_name] = lang
            mapping[repo_name.lower()] = lang
            mapping[key_raw] = lang
            mapping[key_raw.lower()] = lang
            continue

        # Plain folder name
        mapping[key_raw] = lang
        mapping[key_raw.lower()] = lang

    return mapping


def lookup_language(repo_dir: Path, lang_map: dict[str, str]) -> str | None:
    name = repo_dir.name
    return lang_map.get(name) or lang_map.get(name.lower())


def build_info(language: str, graph_hash: str) -> dict[str, Any]:
    return {
        "url": None,
        "description": None,
        "language": language,
        "archived": None,
        "pushed_at": None,
        "stargazers_count": None,
        "hash": graph_hash,
    }


def build_test_dataset(
    repos_dir: Path,
    arcanout_dir: Path,
    lang_map: dict[str, str],
    output_file: Path,
    *,
    max_bytes: int,
    phaseA_depth: int,
    phaseA_max_files: int,
    phaseB_depth: int,
    phaseB_max_files: int,
) -> None:
    per_category_cap = {
        "dependencies": 5,
        "docker": 5,
        "build": 5,
        "readme": 5,
    }

    repos = sorted(
        (p for p in repos_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )

    success = 0
    fail = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as outf:
        pbar = tqdm(repos, desc="Processing repos", unit="repo", dynamic_ncols=True)
        for repo_dir in pbar:
            pbar.set_postfix(ok=success, fail=fail)

            project = repo_dir.name

            # --- Language lookup ---
            language = lookup_language(repo_dir, lang_map)
            if language is None:
                fail += 1
                logging.error("%s: language not found in TSV — skipping", project)
                continue

            # --- Arcan graph ---
            graph_path = arcanout_dir / project
            if not graph_path.is_dir():
                fail += 1
                logging.error(
                    "%s: Arcan output directory not found at %s — skipping",
                    project,
                    graph_path,
                )
                continue

            graph_hash = get_graph_hash(graph_path)
            if graph_hash is None:
                fail += 1
                logging.error(
                    "%s: no .graphml file found in %s — skipping", project, graph_path
                )
                continue

            try:
                package_graph, package_stats = build_graph(
                    project=project,
                    graph_path=graph_path,
                    package_only=True,
                )
                package_files = package_graph.pop("package_files", None)

                original_graph, original_stats = build_graph(
                    project=project,
                    graph_path=graph_path,
                    package_only=False,
                )
                original_graph.pop("package_files", None)

            except Exception:
                fail += 1
                logging.exception("%s: failed to build Arcan graph", project)
                continue

            # --- Context files (inline scan) ---
            try:
                repo_context = scan_repo(
                    repo_dir,
                    language,
                    max_bytes=max_bytes,
                    phaseA_depth=phaseA_depth,
                    phaseA_max_files=phaseA_max_files,
                    phaseB_depth=phaseB_depth,
                    phaseB_max_files=phaseB_max_files,
                    per_category_cap=per_category_cap,
                )
            except Exception:
                logging.exception(
                    "%s: context scan failed, continuing with empty docs", project
                )
                repo_context = {"documents": []}

            result = {
                "repo_id": project,
                "name": project,
                "info": build_info(language, graph_hash),
                "package_stats": package_stats,
                "original_stats": original_stats,
                "package_only_graph": package_graph,
                "original_graph": original_graph,
                "package_files": package_files,
                "context_files": normalize_context_files(
                    repo_context.get("documents", [])
                ),
                "node_compression_ratio": safe_ratio(
                    original_stats["num_nodes"], package_stats["num_nodes"]
                ),
            }

            json.dump(result, outf, ensure_ascii=False)
            outf.write("\n")
            success += 1

        pbar.set_postfix(ok=success, fail=fail)

    logging.info("Done. Success=%d  Fail=%d  Output=%s", success, fail, output_file)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a test/inference dataset from local repos + Arcan output. "
            "Produces the same JSONL structure as combine_arcan_w_context.py "
            "but does not require GitHub metadata or a pre-extracted context file."
        )
    )
    ap.add_argument(
        "--repos_dir",
        required=True,
        help="Directory whose immediate subdirectories are the repos to process.",
    )
    ap.add_argument(
        "--arcanout_dir",
        required=True,
        help="Directory whose subdirectories contain Arcan analysis output per repo.",
    )
    ap.add_argument(
        "--language_tsv",
        required=True,
        help=(
            "TSV file mapping repo names to languages. "
            "First column: folder name, GitHub URL, or owner|repo. "
            "Second column: language (C, CPP, JAVA, PYTHON, CSHARP)."
        ),
    )
    ap.add_argument("--output_file", required=True, help="Output JSONL file path.")
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

    repos_dir = Path(args.repos_dir).expanduser().resolve()
    arcanout_dir = Path(args.arcanout_dir).expanduser().resolve()
    tsv_path = Path(args.language_tsv).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    for p, label in [
        (repos_dir, "--repos_dir"),
        (arcanout_dir, "--arcanout_dir"),
        (tsv_path, "--language_tsv"),
    ]:
        if not p.exists():
            raise SystemExit(f"{label}: path does not exist: {p}")

    lang_map = load_language_tsv(tsv_path)

    build_test_dataset(
        repos_dir=repos_dir,
        arcanout_dir=arcanout_dir,
        lang_map=lang_map,
        output_file=output_file,
        max_bytes=args.max_bytes,
        phaseA_depth=args.phaseA_depth,
        phaseA_max_files=args.phaseA_max_files,
        phaseB_depth=args.phaseB_depth,
        phaseB_max_files=args.phaseB_max_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
