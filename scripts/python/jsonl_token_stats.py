"""
Compute per-repo token counts for each rendered section of the dataset JSONL.

Output: long/tidy format suited for ggplot2.

Columns:
  repo_id,                    -- repo identifiers
  num_packages,               -- number of packages (shared across graph types)
  num_file_package,           -- total files across all packages (shared)
  graph_type,                 -- "package_only" | "original"
  num_nodes,                  -- raw node count
  num_edges,                  -- raw edge count
  nodes_tokens,               -- rendered nodes section
  edges_tokens,               -- rendered edges section
  context_files_tokens,       -- all context files rendered together (shared)
  ctx_readme_tokens,          -- context files of category "readme"
  ctx_docker_tokens,          -- context files of category "docker"
  ctx_dependencies_tokens,    -- context files of category "dependencies"
  ctx_build_tokens,           -- context files of category "build"
  ctx_by_type_total_tokens,   -- sum of per-type token counts
  package_files_tokens,       -- same for both graph types (shared)
  graph_tokens,               -- nodes + edges
  total_tokens                -- nodes + edges + context + package_files
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import tiktoken
from tqdm import tqdm

from reverse_se.generation.utils import DEFAULT_PROMPT_DIR, render_template

PROMPT_DIR = DEFAULT_PROMPT_DIR
ENCODING = tiktoken.encoding_for_model("gpt-5-mini")
CTX_CATEGORIES = ("readme", "docker", "dependencies", "build")


def _tok(text: str) -> int:
    return len(ENCODING.encode(text))


def _rows(repo: dict) -> list[dict]:
    pkg_graph = repo.get("package_only_graph") or {}
    orig_graph = repo.get("original_graph") or {}
    context_files = repo.get("context_files") or []
    package_files = repo.get("package_files") or []

    ctx_tok = _tok(render_template(PROMPT_DIR, "context", context_files=context_files))
    pkg_files_tok = _tok(
        render_template(PROMPT_DIR, "package", package_files=package_files)
    )
    num_file_package = sum(len(p.get("files", [])) for p in package_files)

    ctx_by_cat = {
        cat: _tok(
            render_template(
                PROMPT_DIR,
                "context",
                context_files=[f for f in context_files if f.get("category") == cat],
            )
        )
        for cat in CTX_CATEGORIES
    }

    base = {
        "repo_id": repo.get("repo_id"),
        "num_packages": len(package_files),
        "num_file_package": num_file_package,
        "context_files_tokens": ctx_tok,
        **{f"ctx_{cat}_tokens": ctx_by_cat[cat] for cat in CTX_CATEGORIES},
        "ctx_by_type_total_tokens": sum(ctx_by_cat.values()),
        "package_files_tokens": pkg_files_tok,
    }

    rows = []
    for graph_type, graph in (("package_only", pkg_graph), ("original", orig_graph)):
        nodes_tok = _tok(render_template(PROMPT_DIR, "node", graph=graph))
        edges_tok = _tok(render_template(PROMPT_DIR, "edge", graph=graph))
        num_nodes = len(graph.get("nodes") or [])
        num_edges = len(graph.get("edges") or [])
        rows.append(
            {
                **base,
                "graph_type": graph_type,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "nodes_tokens": nodes_tok,
                "edges_tokens": edges_tok,
                "graph_tokens": nodes_tok + edges_tok,
                "total_tokens": nodes_tok + edges_tok + ctx_tok + pkg_files_tok,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Token stats for dataset JSONL.")
    parser.add_argument("input", type=Path, help="Path to .jsonl file")
    parser.add_argument("output", type=Path, help="Path for output CSV")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows")
    args = parser.parse_args()

    results = []
    errors = 0

    with open(args.input) as fh:
        lines = fh if args.limit is None else (next(fh) for _ in range(args.limit))
        for line in tqdm(lines, desc="Repos", unit="repo", total=args.limit):
            line = line.strip()
            if not line:
                continue
            try:
                repo = json.loads(line)
                results.extend(_rows(repo))
            except Exception as e:
                errors += 1
                print(f"[warn] skipped line: {e}", file=sys.stderr)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Done. Rows={len(df)}, Errors={errors} → {args.output}")


if __name__ == "__main__":
    main()
