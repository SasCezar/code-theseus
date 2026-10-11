#!/usr/bin/env python3
"""Stratified sample of the dataset by language x repo-size x popularity.

A plain random sample tracks the *count* proportions of the population, but
with heavy-tailed variables like stars and LOC it still under-samples the
tails (a handful of huge/very-popular repos get lost among thousands of
small/median ones). This draws a proportionate stratified sample over
(language, size_bin, star_bin) instead, so every combination present in the
population is represented in roughly its population share, with a floor so
rare combinations (e.g. C# / large) aren't dropped to zero by rounding.

Size bins are node-count-based (`original_stats.num_nodes`: classes + impl
files + packages in the full, uncollapsed Arcan graph) rather than LOC, to
match the size proxy the paper itself reports and the metric this dataset
was already percentile-filtered on at build time (see
combine_arcan_w_context.py's p10-95 node-count filter). There is no single
canonical node/file-count size scheme the way Ohloh/Open Hub's is for LOC,
but software-clustering / architecture-recovery benchmark literature (e.g.
Wu, Hassan & Holt-style clustering evaluations) commonly groups subject
systems by module/class/file count using the same order-of-magnitude logic
as LOC-based schemes, just ~2 orders of magnitude down given ~100 LOC/file
on average:
    small   : < 100 nodes
    medium  : 100 - 1,000 nodes
    large   : >= 1,000 nodes
These round numbers also split this (already p10-95-filtered) population
close to its natural quantiles: 51% small / 45% medium / 4% large.

Star bins are round-number popularity brackets:
    low     : < 500 stars
    medium  : 500 - 5,000 stars
    high    : >= 5,000 stars

Runs in two streaming passes over the (large) source file so the full
dataset is never held in memory at once: pass 1 parses each line only to
compute its stratum; pass 2 re-reads the file and copies selected lines
verbatim to the output.

Usage:
    uv run scripts/python/stratified_sample.py
    uv run scripts/python/stratified_sample.py --n 1000 --seed 42
    uv run scripts/python/stratified_sample.py --min-nodes 100 --max-nodes 1000
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import ROOT_DIR

DEFAULT_INPUT = ROOT_DIR / "data/out/reverse-se_v0.6_p10-95_ctx95_dataset.jsonl"
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000_stratified.jsonl"
)

SIZE_BIN_EDGES = [0, 100, 1_000, float("inf")]
SIZE_BIN_LABELS = ["small", "medium", "large"]

STAR_BIN_EDGES = [0, 500, 5_000, float("inf")]
STAR_BIN_LABELS = ["low", "medium", "high"]


def bin_value(value: float, edges: list[float], labels: list[str]) -> str:
    for lo, hi, label in zip(edges[:-1], edges[1:], labels, strict=False):
        if lo <= value < hi:
            return label
    return labels[-1]


def repo_num_files(record: dict) -> int:
    return sum(len(p.get("files", [])) for p in record.get("package_files") or [])


def scan_strata(input_path: Path) -> pd.DataFrame:
    """Pass 1: stream the file and compute each repo's stratum (no full record retained)."""
    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, desc="Scanning", unit="repo")):
            d = json.loads(line)
            stars = d["info"].get("stargazers_count") or 0
            language = d["info"].get("language") or "UNKNOWN"
            num_nodes = d["original_stats"]["num_nodes"]
            num_files = repo_num_files(d)
            rows.append(
                {
                    "line_no": line_no,
                    "repo_id": d["repo_id"],
                    "language": language,
                    "stars": stars,
                    "num_nodes": num_nodes,
                    "num_files": num_files,
                    "size_bin": bin_value(num_nodes, SIZE_BIN_EDGES, SIZE_BIN_LABELS),
                    "star_bin": bin_value(stars, STAR_BIN_EDGES, STAR_BIN_LABELS),
                }
            )
    df = pd.DataFrame(rows)
    df["size_bin"] = pd.Categorical(
        df["size_bin"], categories=SIZE_BIN_LABELS, ordered=True
    )
    df["star_bin"] = pd.Categorical(
        df["star_bin"], categories=STAR_BIN_LABELS, ordered=True
    )
    return df


def allocate_stratum_sizes(
    counts: pd.Series, n: int, min_per_stratum: int, method: str = "sqrt"
) -> pd.Series:
    """Largest-remainder allocation across strata in `counts`, every non-empty
    stratum floored at min(min_per_stratum, count).

    method="proportional" allocates the remaining budget in proportion to
    stratum population, so the sample mirrors the population's real-world
    skew (e.g. small/low-star/Python dominates, large/rare combos get ~1-6).
    method="sqrt" (default) allocates proportional to sqrt(population)
    instead, which compresses that skew: common strata still get more
    samples than rare ones, but not in lockstep with how rare they are.
    This gives every stratum enough repos for a meaningful per-stratum
    comparison, which matters for a benchmark meant to evaluate performance
    *by* size/language/stars rather than just describe the population.
    """
    population = int(counts.sum())
    if n > population:
        raise SystemExit(f"--n={n} exceeds the population size ({population}).")

    floors = counts.clip(upper=min_per_stratum)
    remaining = n - int(floors.sum())
    if remaining < 0:
        raise SystemExit(
            f"--n={n} is smaller than {len(counts)} strata x --min-per-stratum="
            f"{min_per_stratum} ({int(floors.sum())}); raise --n or lower --min-per-stratum."
        )

    capacity = counts - floors
    allocated = floors.copy()

    if remaining and capacity.sum() > 0:
        weight = np.sqrt(capacity) if method == "sqrt" else capacity
        ideal_extra = remaining * weight / weight.sum()
        extra = np.floor(ideal_extra).astype(int).clip(upper=capacity)
        allocated += extra
        leftover = remaining - int(extra.sum())

        remainder_order = (ideal_extra - extra).sort_values(ascending=False).index
        for stratum in itertools.cycle(remainder_order):
            if leftover == 0:
                break
            if allocated[stratum] < counts[stratum]:
                allocated[stratum] += 1
                leftover -= 1

    assert int(allocated.sum()) == n
    return allocated


def select_line_numbers(
    df: pd.DataFrame, allocated: pd.Series, rng: random.Random
) -> set[int]:
    selected: set[int] = set()
    for stratum_key, group in df.groupby(
        ["language", "size_bin", "star_bin"], observed=True
    ):
        k = int(allocated.loc[stratum_key])
        if k > 0:
            selected.update(rng.sample(list(group["line_no"]), k))
    return selected


def write_sample(
    input_path: Path, output_path: Path, selected_line_nos: set[int]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        input_path.open("r", encoding="utf-8") as inf,
        output_path.open("w", encoding="utf-8") as outf,
    ):
        for line_no, line in enumerate(tqdm(inf, desc="Writing sample", unit="repo")):
            if line_no in selected_line_nos:
                outf.write(line if line.endswith("\n") else line + "\n")
                written += 1
    print(f"Wrote {written} repos -> {output_path}")


def build_strata_table(
    counts: pd.Series, allocated: pd.Series, population: int, n: int
) -> pd.DataFrame:
    stats = pd.DataFrame({"population": counts, "sample": allocated}).sort_index()
    stats["population_pct"] = (100 * stats["population"] / population).round(2)
    stats["sample_pct"] = (100 * stats["sample"] / n).round(2)
    return stats[["population", "population_pct", "sample", "sample_pct"]]


def print_marginal(
    label: str, df: pd.DataFrame, sample_mask: pd.Series, col: str
) -> None:
    pop = df.groupby(col, observed=True).size().sort_index()
    samp = df.loc[sample_mask].groupby(col, observed=True).size().sort_index()
    out = pd.DataFrame({"population": pop, "sample": samp}).fillna(0).astype(int)
    out["population_pct"] = (100 * out["population"] / out["population"].sum()).round(1)
    out["sample_pct"] = (100 * out["sample"] / out["sample"].sum()).round(1)
    print(f"\n{label}:")
    print(out[["population", "population_pct", "sample", "sample_pct"]].to_string())


def print_metric_by_size_bin(
    df: pd.DataFrame, sample_mask: pd.Series, col: str, label: str
) -> None:
    agg = ["count", "min", "median", "mean", "max"]
    pop = df.groupby("size_bin", observed=True)[col].agg(agg).round(1)
    samp = df.loc[sample_mask].groupby("size_bin", observed=True)[col].agg(agg).round(1)
    out = pop.join(samp, lsuffix="_population", rsuffix="_sample")
    print(f"\n{label} by size_bin:")
    print(out.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-per-stratum",
        type=int,
        default=1,
        help="Minimum samples guaranteed for every non-empty "
        "(language, size_bin, star_bin) stratum",
    )
    parser.add_argument(
        "--allocation",
        choices=["sqrt", "proportional"],
        default="sqrt",
        help="How to split the remaining budget across strata beyond the floor: "
        "'sqrt' (default) balances coverage across strata, 'proportional' "
        "mirrors the population's real-world skew",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Default: <output stem>.strata_stats.csv next to --output",
    )
    parser.add_argument(
        "--min-nodes",
        type=float,
        default=None,
        help="Restrict the population to num_nodes >= this before stratifying "
        "(e.g. 100 to drop the 'small' bin entirely). Unlike size_bin, which "
        "only labels repos, this actually excludes them from the population "
        "and from every stratum count/marginal printed below.",
    )
    parser.add_argument(
        "--max-nodes",
        type=float,
        default=None,
        help="Restrict the population to num_nodes <= this before stratifying "
        "(e.g. 1000 to drop the 'large' bin entirely).",
    )
    args = parser.parse_args()
    stats_output = args.stats_output or args.output.with_name(
        args.output.stem + ".strata_stats.csv"
    )

    print(f"Scanning {args.input} ...")
    df = scan_strata(args.input)
    print(f"Population (pre node-count filter): {len(df)} repos")

    if args.min_nodes is not None:
        df = df[df["num_nodes"] >= args.min_nodes]
    if args.max_nodes is not None:
        df = df[df["num_nodes"] <= args.max_nodes]
    df = df.reset_index(drop=True)
    population = len(df)
    if args.min_nodes is not None or args.max_nodes is not None:
        print(
            f"Population (num_nodes in [{args.min_nodes}, {args.max_nodes}]): "
            f"{population} repos"
        )

    counts = df.groupby(["language", "size_bin", "star_bin"], observed=True).size()
    allocated = allocate_stratum_sizes(
        counts, args.n, args.min_per_stratum, method=args.allocation
    )

    rng = random.Random(args.seed)
    selected_line_nos = select_line_numbers(df, allocated, rng)
    assert len(selected_line_nos) == args.n

    write_sample(args.input, args.output, selected_line_nos)

    strata_table = build_strata_table(counts, allocated, population, args.n)
    print("\nPer-stratum (language x size_bin x star_bin):")
    print(strata_table.to_string())
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    strata_table.to_csv(stats_output)
    print(f"\nStrata stats -> {stats_output}")

    sample_mask = df["line_no"].isin(selected_line_nos)
    print_marginal("By language", df, sample_mask, "language")
    print_marginal("By size_bin", df, sample_mask, "size_bin")
    print_marginal("By star_bin", df, sample_mask, "star_bin")
    print_metric_by_size_bin(
        df, sample_mask, "num_nodes", "num_nodes (classes + files + packages)"
    )
    print_metric_by_size_bin(
        df, sample_mask, "num_files", "num_files (files across all packages)"
    )


if __name__ == "__main__":
    main()
