#!/usr/bin/env python3
"""
repo_token_count.py — Count source-file tokens for a repo and optionally compare
with the tool's JSONL output.

Usage:
  python repo_token_count.py --repo-path /path/to/repo [options]

Options:
  --language LANG         java|python|c|cpp|csharp — restrict walk to that language's
                          extensions (ignored when --arcan-output is given)
  --arcan-output PATH     Arcan CSV output dir; uses component-metrics.csv to get
                          the exact set of files Arcan analysed
  --dataset-jsonl PATH    Dataset JSONL produced by the tool; prints a comparison
  --repo-id NAME          Repo ID to look up in the JSONL (default: repo-path basename)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import tiktoken

from reverse_se.generation.utils import DEFAULT_PROMPT_DIR, render_template

_PROMPT_DIR = DEFAULT_PROMPT_DIR

_ENCODING = tiktoken.encoding_for_model("gpt-5-mini")

# ── Extension sets ─────────────────────────────────────────────────────────────

_LANG_EXTENSIONS: dict[str, frozenset[str]] = {
    "java": frozenset({".java"}),
    "python": frozenset({".py"}),
    "c": frozenset({".c", ".h"}),
    "cpp": frozenset({".cpp", ".cc", ".cxx", ".h", ".hpp"}),
    "csharp": frozenset({".cs"}),
}

_ALL_EXTENSIONS: frozenset[str] = frozenset(
    ext for exts in _LANG_EXTENSIONS.values() for ext in exts
) | frozenset({".kt", ".scala", ".go"})

_SKIP_DIRS: frozenset[str] = frozenset(
    {
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
)


# ── File list helpers ──────────────────────────────────────────────────────────


def files_from_arcan(arcan_output: Path) -> list[str]:
    """Return filePathRelative values for all unit (non-container) nodes."""
    metrics = arcan_output / "component-metrics.csv"
    if not metrics.exists():
        raise SystemExit(f"component-metrics.csv not found in {arcan_output}")
    paths: list[str] = []
    with open(metrics, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("vertexLabel") == "unit":
                rel = row.get("filePathRelative", "").strip()
                if rel:
                    paths.append(rel)
    return paths


def files_from_walk(repo_path: Path, extensions: frozenset[str]) -> list[str]:
    """Walk repo_path (preferring src/ if present), return relative paths."""
    start = repo_path / "src"
    if not start.is_dir():
        start = repo_path
    paths: list[str] = []
    for root, dirs, files in os.walk(start):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        root_p = Path(root)
        for fname in files:
            if Path(fname).suffix.lower() in extensions:
                paths.append((root_p / fname).relative_to(repo_path).as_posix())
    return paths


# ── Token counting ─────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


def count_repo_tokens(
    repo_path: Path, rel_paths: list[str]
) -> tuple[int, dict[str, int], int]:
    """
    Count tokens for each file. Returns (total, per_extension, missing_count).
    """
    per_ext: dict[str, int] = {}
    total = 0
    missing = 0
    for rel in rel_paths:
        abs_path = repo_path / rel
        if not abs_path.exists():
            missing += 1
            continue
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            missing += 1
            continue
        toks = count_tokens(text)
        ext = Path(rel).suffix.lower() or "(no ext)"
        per_ext[ext] = per_ext.get(ext, 0) + toks
        total += toks
    return total, per_ext, missing


# ── JSONL comparison ───────────────────────────────────────────────────────────


def find_repo_in_jsonl(jsonl_path: Path, repo_id: str) -> dict | None:
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("repo_id") == repo_id or record.get("name") == repo_id:
                return record
    return None


def _render_graph_tokens(graph: dict) -> tuple[int, int]:
    """Return (nodes_tokens, edges_tokens) for a rendered graph."""
    return (
        count_tokens(render_template(_PROMPT_DIR, "node", graph=graph)),
        count_tokens(render_template(_PROMPT_DIR, "edge", graph=graph)),
    )


def tool_tokens(record: dict) -> dict:
    """Render all sections; return token counts keyed by section name."""
    context_files = record.get("context_files") or []
    package_files = record.get("package_files") or []
    pkg_nodes, pkg_edges = _render_graph_tokens(record.get("package_only_graph") or {})
    raw_nodes, raw_edges = _render_graph_tokens(record.get("original_graph") or {})
    ctx = count_tokens(
        render_template(_PROMPT_DIR, "context", context_files=context_files)
    )
    pkg_files = count_tokens(
        render_template(_PROMPT_DIR, "package", package_files=package_files)
    )
    return {
        "ctx": ctx,
        "pkg_files": pkg_files,
        "pkg_nodes": pkg_nodes,
        "pkg_edges": pkg_edges,
        "raw_nodes": raw_nodes,
        "raw_edges": raw_edges,
    }


# ── Formatting ─────────────────────────────────────────────────────────────────

_W = 42
_SEP = "─" * _W


def _fmt(n: int) -> str:
    return f"{n:,}"


def print_source_stats(
    repo_id: str,
    mode: str,
    file_count: int,
    missing: int,
    per_ext: dict[str, int],
    total: int,
) -> None:
    print(f"\nSource file token count for: {repo_id}")
    print(f"Mode         : {mode}")
    print(f"Files found  : {_fmt(file_count)}")
    if missing:
        print(f"Files missing: {_fmt(missing)}", file=sys.stderr)
    print(_SEP)
    for ext, toks in sorted(per_ext.items(), key=lambda x: -x[1]):
        print(f"  {ext:<12} {_fmt(toks):>16}")
    print(_SEP)
    print(f"  {'TOTAL':<12} {_fmt(total):>16}")


def _pct(part: int, total: int) -> str:
    return f"({part / total * 100:5.1f} %)" if total else "(  n/a )"


def print_comparison(source_total: int, t: dict) -> None:
    pkg_total = t["pkg_nodes"] + t["pkg_edges"] + t["pkg_files"] + t["ctx"]
    raw_total = t["raw_nodes"] + t["raw_edges"] + t["pkg_files"] + t["ctx"]
    print(f"\n{_SEP}")
    print(f"  {'':20} {'package_only':>12}  {'original':>12}")
    print(_SEP)
    print(
        f"  {'Source files':20} {_fmt(source_total):>12}  {_fmt(source_total):>12}  (100.0 %)"
    )
    print(
        f"  {'Tool total':20} {_fmt(pkg_total):>12}  {_fmt(raw_total):>12}  "
        f"{_pct(pkg_total, source_total)} / {_pct(raw_total, source_total)}"
    )
    print(_SEP)
    print(f"  {'  nodes':20} {_fmt(t['pkg_nodes']):>12}  {_fmt(t['raw_nodes']):>12}")
    print(f"  {'  edges':20} {_fmt(t['pkg_edges']):>12}  {_fmt(t['raw_edges']):>12}")
    print(
        f"  {'  package files':20} {_fmt(t['pkg_files']):>12}  {_fmt(t['pkg_files']):>12}"
    )
    print(f"  {'  context files':20} {_fmt(t['ctx']):>12}  {_fmt(t['ctx']):>12}")
    print(_SEP)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Count source-file tokens for a repo, optionally comparing with tool output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo-path", required=True, help="Path to the repository root.")
    ap.add_argument(
        "--language",
        choices=list(_LANG_EXTENSIONS),
        default=None,
        help="Restrict walk to this language's extensions (ignored with --arcan-output).",
    )
    ap.add_argument(
        "--arcan-output",
        default=None,
        help="Arcan CSV output directory (component-metrics.csv must be present).",
    )
    ap.add_argument(
        "--dataset-jsonl",
        default=None,
        help="Dataset JSONL produced by the tool; enables comparison output.",
    )
    ap.add_argument(
        "--repo-id",
        default=None,
        help="Repo ID to look up in the JSONL (default: basename of --repo-path).",
    )
    args = ap.parse_args()

    repo_path = Path(args.repo_path).expanduser().resolve()
    if not repo_path.is_dir():
        raise SystemExit(f"--repo-path does not exist: {repo_path}")

    repo_id = args.repo_id or repo_path.name

    if args.arcan_output:
        arcan_output = Path(args.arcan_output).expanduser().resolve()
        rel_paths = files_from_arcan(arcan_output)
        mode = f"arcan ({arcan_output.name})"
    else:
        extensions = _LANG_EXTENSIONS.get(args.language or "", _ALL_EXTENSIONS)
        if args.language:
            extensions = _LANG_EXTENSIONS[args.language]
        rel_paths = files_from_walk(repo_path, extensions)
        mode = f"walk (lang={args.language or 'all'})"

    total, per_ext, missing = count_repo_tokens(repo_path, rel_paths)
    print_source_stats(repo_id, mode, len(rel_paths), missing, per_ext, total)

    if args.dataset_jsonl:
        jsonl_path = Path(args.dataset_jsonl).expanduser().resolve()
        if not jsonl_path.exists():
            print(f"\n[warn] --dataset-jsonl not found: {jsonl_path}", file=sys.stderr)
            return 0
        record = find_repo_in_jsonl(jsonl_path, repo_id)
        if record is None:
            print(
                f"\n[warn] repo_id '{repo_id}' not found in {jsonl_path}",
                file=sys.stderr,
            )
            return 0
        print_comparison(total, tool_tokens(record))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
