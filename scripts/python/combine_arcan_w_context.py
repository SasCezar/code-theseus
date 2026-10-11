import argparse
import csv
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypedDict

import tiktoken
from tqdm import tqdm

from reverse_se.graph.arcan import ArcanCSV

_enc = tiktoken.encoding_for_model("gpt-5-mini")

# --- Constants ---

_DEFAULT_P_LOW = 5.0
_DEFAULT_P_HIGH = 95.0
_DEFAULT_CTX_P_HIGH = 95.0
_DEFAULT_WORKERS = 1

_NODE_RENAME: dict[str, str] = {
    "ComponentType": "component_type",
    "FanIn": "fan_in",
    "FanOut": "fan_out",
    "LinesOfCode": "lines_of_code",
    "PageRank": "pagerank",
    "name": "name",
    "vertexId": "vertex_id",
}

_EDGE_RENAME: dict[str, str] = {
    "fromId": "source",
    "toId": "target",
    "Weight": "weight",
}

_STATS_CSV_FIELDS = [
    "repo",
    "package_num_nodes",
    "package_num_edges",
    "package_num_packages",
    "original_num_nodes",
    "original_num_edges",
    "original_num_packages",
    "node_compression_ratio",
    "edge_compression_ratio",
    "filtered",
]


# --- Types ---


class GraphStats(TypedDict):
    num_nodes: int
    num_edges: int
    num_packages: int


class ContextFileRecord(TypedDict):
    path: str | None
    category: str | None
    content: str | None
    tokens: int


class RepoResult(TypedDict):
    repo_id: str
    name: str
    info: dict[str, Any]
    package_stats: GraphStats
    original_stats: GraphStats
    package_only_graph: dict[str, Any]
    original_graph: dict[str, Any]
    package_files: list[dict[str, Any]] | None
    context_files: list[ContextFileRecord]
    node_compression_ratio: float
    edge_compression_ratio: float


class RepoStats(TypedDict):
    repo: str
    package_num_nodes: int
    package_num_edges: int
    package_num_packages: int
    original_num_nodes: int
    original_num_edges: int
    original_num_packages: int
    node_compression_ratio: float
    edge_compression_ratio: float
    filtered: bool


# --- Utilities ---


def safe_ratio(raw: int | float, compressed: int | float) -> float:
    if compressed <= 0:
        return float("inf")
    return float(raw) / float(compressed)


def compute_percentile(values: list[int | float], p: float) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    idx = p / 100 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def count_tokens(content: str | None) -> int:
    if not content:
        return 0
    return len(_enc.encode(content, disallowed_special=()))


def _to_builtin_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _clean_value(value: Any) -> Any:
    value = _to_builtin_scalar(value)
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


# --- Data normalization ---


def _dataframe_to_records(df) -> list[dict[str, Any]]:
    if df is None or len(df) == 0:
        return []
    return [
        {str(k): _clean_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def normalize_nodes(nodes_df) -> list[dict[str, Any]]:
    if nodes_df is None or len(nodes_df) == 0:
        return []
    available = [c for c in _NODE_RENAME if c in nodes_df.columns]
    return _dataframe_to_records(nodes_df[available].rename(columns=_NODE_RENAME))


def normalize_edges(edges_df) -> list[dict[str, Any]]:
    if edges_df is None or len(edges_df) == 0:
        return []
    available = [c for c in _EDGE_RENAME if c in edges_df.columns]
    return _dataframe_to_records(edges_df[available].rename(columns=_EDGE_RENAME))


def normalize_package_files(package_files_df) -> list[dict[str, Any]]:
    if package_files_df is None or len(package_files_df) == 0:
        return []
    return [
        {"package": package, "files": files}
        for package, files in package_files_df.itertuples(index=False)
    ]


def _normalize_context_file(doc: dict[str, Any]) -> ContextFileRecord:
    content = doc.get("content")
    return {
        "path": doc.get("path"),
        "category": doc.get("category"),
        "content": content,
        "tokens": doc.get("tokens") or count_tokens(content),
    }


def normalize_context_files(documents: list[dict[str, Any]]) -> list[ContextFileRecord]:
    return list(map(_normalize_context_file, documents))


def filter_context_files(
    context_files: list[ContextFileRecord],
    thresholds: dict[str, float],
) -> tuple[list[ContextFileRecord], int]:
    kept, dropped = [], 0
    for f in context_files:
        threshold = thresholds.get(f.get("category") or "")
        if threshold is not None and f.get("tokens", 0) > threshold:
            dropped += 1
        else:
            kept.append(f)
    return kept, dropped


# --- Graph building ---


def get_graph_hash(graph_path: Path) -> str | None:
    hash_file = next(graph_path.glob("*.graphml"), None)
    if hash_file is None:
        return None
    parts = hash_file.stem.split("_", 1)
    return parts[1] if len(parts) > 1 else hash_file.stem


def build_graph(
    project: str,
    graph_path: Path,
    package_only: bool,
) -> tuple[dict[str, Any], GraphStats]:
    nodes, edges, package_files = ArcanCSV(
        name=project,
        path=graph_path,
        package_only=package_only,
    ).graph

    normalized_nodes = normalize_nodes(nodes)
    normalized_edges = normalize_edges(edges)
    normalized_package_files = normalize_package_files(package_files)

    stats: GraphStats = {
        "num_nodes": len(normalized_nodes),
        "num_edges": len(normalized_edges),
        "num_packages": len(normalized_package_files),
    }
    graph = {
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "package_files": normalized_package_files,
    }
    return graph, stats


def _normalize_info(info: dict[str, Any] | None, graph_hash: str) -> dict[str, Any]:
    info = dict(info or {})
    info.pop("name", None)
    return {
        "url": info.get("url"),
        "description": info.get("description"),
        "language": info.get("language"),
        "archived": info.get("archived"),
        "pushed_at": info.get("pushed_at"),
        "stargazers_count": info.get("stargazers_count"),
        "hash": graph_hash,
    }


# --- Per-repo processing ---


def load_info_map(info_file_path: str | Path) -> dict[str, dict[str, Any]]:
    with open(info_file_path, encoding="utf-8") as f:
        return {row["name"]: row for row in map(json.loads, f)}


def process_repo(
    repo_context: dict[str, Any],
    arcanout_dir: Path,
    info_map: dict[str, dict[str, Any]],
) -> tuple[RepoResult, RepoStats] | None:
    """Build the full representation for one repo. Returns None if the graph is missing."""
    project = repo_context["repo"].strip()
    graph_path = arcanout_dir / project

    graph_hash = get_graph_hash(graph_path)
    if graph_hash is None:
        return None

    package_graph, package_stats = build_graph(project, graph_path, package_only=True)
    package_files = package_graph.pop("package_files", None)

    original_graph, original_stats = build_graph(
        project, graph_path, package_only=False
    )
    original_graph.pop("package_files", None)

    context_files = normalize_context_files(repo_context.get("documents", []))
    node_compression_ratio = safe_ratio(
        original_stats["num_nodes"], package_stats["num_nodes"]
    )
    edge_compression_ratio = safe_ratio(
        original_stats["num_edges"], package_stats["num_edges"]
    )

    result: RepoResult = {
        "repo_id": project,
        "name": project,
        "info": _normalize_info(info_map.get(project), graph_hash),
        "package_stats": package_stats,
        "original_stats": original_stats,
        "package_only_graph": package_graph,
        "original_graph": original_graph,
        "package_files": package_files,
        "context_files": context_files,
        "node_compression_ratio": node_compression_ratio,
        "edge_compression_ratio": edge_compression_ratio,
    }

    stats: RepoStats = {
        "repo": project,
        "package_num_nodes": package_stats["num_nodes"],
        "package_num_edges": package_stats["num_edges"],
        "package_num_packages": package_stats["num_packages"],
        "original_num_nodes": original_stats["num_nodes"],
        "original_num_edges": original_stats["num_edges"],
        "original_num_packages": original_stats["num_packages"],
        "node_compression_ratio": node_compression_ratio,
        "edge_compression_ratio": edge_compression_ratio,
        "filtered": True,
    }

    return result, stats


# --- Dataset assembly ---


def combine_dataset(
    arcanout_dir_path: str | Path,
    context_file_path: str | Path,
    info_file_path: str | Path,
    output_file_path: str | Path,
    p_low: float = _DEFAULT_P_LOW,
    p_high: float = _DEFAULT_P_HIGH,
    ctx_p_high: float = _DEFAULT_CTX_P_HIGH,
    node_compression_p_high: float | None = None,
    edge_compression_p_high: float | None = None,
    workers: int = _DEFAULT_WORKERS,
    stats_file_path: str | Path | None = None,
) -> None:
    arcanout_dir = Path(arcanout_dir_path)
    info_map = load_info_map(info_file_path)

    success = 0
    fail = 0
    num_nodes_list: list[int | float] = []
    node_compression_ratios: list[int | float] = []
    edge_compression_ratios: list[int | float] = []
    ctx_tokens_by_category: dict[str, list[int | float]] = {}
    repo_stats: dict[str, RepoStats] = {}

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(tmp_fd)

    try:
        # Pass 1: process all repos into a temp file; collect stats for percentile computation.
        with (
            open(context_file_path, encoding="utf-8") as contextf,
            open(tmp_path, "w", encoding="utf-8") as tmpf,
            ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            lines = list(contextf)
            futures = {
                executor.submit(
                    process_repo, json.loads(line), arcanout_dir, info_map
                ): line
                for line in lines
            }

            pbar = tqdm(
                as_completed(futures), total=len(futures), desc="Processing repos"
            )
            for future in pbar:
                try:
                    outcome = future.result()
                except Exception as e:
                    fail += 1
                    tqdm.write(str(e))
                    pbar.set_postfix(ok=success, fail=fail)
                    continue

                if outcome is None:
                    fail += 1
                    pbar.set_postfix(ok=success, fail=fail)
                    continue

                result, stats = outcome
                project = result["repo_id"]

                num_nodes_list.append(result["original_stats"]["num_nodes"])
                node_compression_ratios.append(result["node_compression_ratio"])
                edge_compression_ratios.append(result["edge_compression_ratio"])

                for cf in result["context_files"]:
                    ctx_tokens_by_category.setdefault(
                        cf.get("category", ""), []
                    ).append(cf.get("tokens", 0))

                repo_stats[project] = stats
                json.dump(result, tmpf, ensure_ascii=False)
                tmpf.write("\n")

                success += 1
                pbar.set_postfix(ok=success, fail=fail)

        # Compute node-count percentile bounds
        p5 = compute_percentile(num_nodes_list, p_low)
        p95 = compute_percentile(num_nodes_list, p_high)
        print(f"\nOriginal graph node count — p{p_low}={p5:.1f}, p{p_high}={p95:.1f}")

        # Compute optional node-compression threshold
        node_compression_threshold: float | None = None
        if node_compression_p_high is not None:
            finite_ratios = [r for r in node_compression_ratios if r != float("inf")]
            node_compression_threshold = compute_percentile(
                finite_ratios, node_compression_p_high
            )
            print(
                f"Node compression ratio — p{node_compression_p_high}={node_compression_threshold:.2f}x"
            )

        # Compute optional edge-compression threshold
        edge_compression_threshold: float | None = None
        if edge_compression_p_high is not None:
            finite_ratios = [r for r in edge_compression_ratios if r != float("inf")]
            edge_compression_threshold = compute_percentile(
                finite_ratios, edge_compression_p_high
            )
            print(
                f"Edge compression ratio — p{edge_compression_p_high}={edge_compression_threshold:.2f}x"
            )

        # Compute per-category context-file token thresholds
        ctx_thresholds: dict[str, float] = {
            cat: compute_percentile(tokens, ctx_p_high)
            for cat, tokens in ctx_tokens_by_category.items()
        }
        print(f"Context file token thresholds (p{ctx_p_high} per category):")
        for cat, threshold in sorted(ctx_thresholds.items()):
            print(f"  {cat}: {threshold:.0f}")

        # Pass 2: filter by node count / compression, strip oversized context files, write output.
        kept = 0
        total_ctx_dropped = 0
        with (
            open(tmp_path, encoding="utf-8") as tmpf,
            open(output_file_path, "w", encoding="utf-8") as outf,
        ):
            for line in tqdm(tmpf, desc="Filtering"):
                result = json.loads(line)
                n = result["original_stats"]["num_nodes"]
                passes_node_count = p5 <= n <= p95
                passes_node_compression = node_compression_threshold is None or (
                    result["node_compression_ratio"] <= node_compression_threshold
                )
                passes_edge_compression = edge_compression_threshold is None or (
                    result["edge_compression_ratio"] <= edge_compression_threshold
                )

                if (
                    passes_node_count
                    and passes_node_compression
                    and passes_edge_compression
                ):
                    result["context_files"], ctx_dropped = filter_context_files(
                        result["context_files"], ctx_thresholds
                    )
                    total_ctx_dropped += ctx_dropped
                    json.dump(result, outf, ensure_ascii=False)
                    outf.write("\n")
                    repo_stats[result["repo_id"]]["filtered"] = False
                    kept += 1

        filtered_count = success - kept
        print(
            f"Kept {kept}/{success} repos (original node count in [{p5:.1f}, {p95:.1f}]),"
            f" filtered {filtered_count}"
        )
        print(
            f"Dropped {total_ctx_dropped} context files exceeding p{ctx_p_high} token threshold per category"
        )

        if stats_file_path is not None:
            _write_stats_csv(stats_file_path, repo_stats)
            print(f"Stats written to {stats_file_path}")

    finally:
        os.unlink(tmp_path)


def _write_stats_csv(path: str | Path, repo_stats: dict[str, RepoStats]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_STATS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(repo_stats.values())


def _build_stem(
    name: str,
    version: str,
    p_low: float,
    p_high: float,
    ctx_p_high: float,
    node_compression_p_high: float | None,
    edge_compression_p_high: float | None,
) -> str:
    parts = [
        name,
        version,
        f"p{p_low:.0f}-{p_high:.0f}",
        f"ctx{ctx_p_high:.0f}",
    ]
    if node_compression_p_high is not None:
        parts.append(f"ncomp{node_compression_p_high:.0f}")
    if edge_compression_p_high is not None:
        parts.append(f"ecomp{edge_compression_p_high:.0f}")
    return "_".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine Arcan data with context into a single JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Inputs ---
    parser.add_argument(
        "--arcanout_dir", required=True, help="Path to Arcan output directory"
    )
    parser.add_argument("--context_file", required=True)
    parser.add_argument("--info_file", required=True)

    # --- Output naming (auto-generated from hyperparams) ---
    parser.add_argument("--name", default="reverse-se", help="Dataset base name")
    parser.add_argument("--version", required=True, help="Version tag, e.g. v0.3")
    parser.add_argument(
        "--output_file", default=None, help="Override auto-generated output path"
    )
    parser.add_argument(
        "--stats_file", default=None, help="Override auto-generated stats path"
    )

    # --- Filtering hyperparams ---
    parser.add_argument(
        "--p_low",
        type=float,
        default=_DEFAULT_P_LOW,
        help="Lower node-count percentile cutoff",
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=_DEFAULT_P_HIGH,
        help="Upper node-count percentile cutoff",
    )
    parser.add_argument(
        "--ctx_p_high",
        type=float,
        default=_DEFAULT_CTX_P_HIGH,
        help="Per-category token-count percentile cutoff for context files",
    )
    parser.add_argument(
        "--node_compression_p_high",
        type=float,
        default=None,
        help="Upper percentile cutoff for original/package node ratio. "
        "Example: 99 drops the top 1%%.",
    )
    parser.add_argument(
        "--edge_compression_p_high",
        type=float,
        default=None,
        help="Upper percentile cutoff for original/package edge ratio. "
        "Example: 99 drops the top 1%%.",
    )

    # --- Performance ---
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help="Parallel workers for repo processing",
    )

    args = parser.parse_args()

    stem = _build_stem(
        args.name,
        args.version,
        args.p_low,
        args.p_high,
        args.ctx_p_high,
        args.node_compression_p_high,
        args.edge_compression_p_high,
    )
    output_file = args.output_file or f"{stem}_dataset.jsonl"
    stats_file = args.stats_file or f"{stem}_stats.csv"

    print(f"Output : {output_file}")
    print(f"Stats  : {stats_file}")

    combine_dataset(
        args.arcanout_dir,
        args.context_file,
        args.info_file,
        output_file,
        p_low=args.p_low,
        p_high=args.p_high,
        ctx_p_high=args.ctx_p_high,
        node_compression_p_high=args.node_compression_p_high,
        edge_compression_p_high=args.edge_compression_p_high,
        workers=args.workers,
        stats_file_path=stats_file,
    )
