"""Correlate repo-level judge scores Q(m,x) with the sampling strata
dimensions (language, size_bin, star_bin) and their underlying continuous
variables (num_nodes, num_files, stars), to check whether model performance
depends on repo size/popularity/language rather than being uniform.

Size/star bins are recomputed with the same edges as
scripts/python/stratified_sample.py so they match the strata the sample was
drawn on.

Usage:
    uv run python -m reverse_se.evaluation.judge.strata_correlation \\
        --repo_level data/out/judge/aggregate/Q_repo_level.csv \\
        --dataset data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_100_stratified.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats

SIZE_BIN_EDGES = [0, 100, 1_000, float("inf")]
SIZE_BIN_LABELS = ["small", "medium", "large"]

STAR_BIN_EDGES = [0, 500, 5_000, float("inf")]
STAR_BIN_LABELS = ["low", "medium", "high"]

CONTINUOUS_VARS = ["num_nodes", "num_files", "stars"]
CATEGORICAL_VARS = ["language", "size_bin", "star_bin"]


def bin_value(value: float, edges: list[float], labels: list[str]) -> str:
    for lo, hi, label in zip(edges[:-1], edges[1:], labels, strict=False):
        if lo <= value < hi:
            return label
    return labels[-1]


def repo_num_files(record: dict) -> int:
    return sum(len(p.get("files", [])) for p in record.get("package_files") or [])


def load_strata(dataset_path: Path) -> pd.DataFrame:
    rows = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            stars = d["info"].get("stargazers_count") or 0
            num_nodes = d["original_stats"]["num_nodes"]
            rows.append(
                {
                    "repo_id": d["repo_id"],
                    "language": d["info"].get("language") or "UNKNOWN",
                    "stars": stars,
                    "num_nodes": num_nodes,
                    "num_files": repo_num_files(d),
                    "size_bin": bin_value(num_nodes, SIZE_BIN_EDGES, SIZE_BIN_LABELS),
                    "star_bin": bin_value(stars, STAR_BIN_EDGES, STAR_BIN_LABELS),
                }
            )
    return pd.DataFrame(rows)


def continuous_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman correlation (with p-values) of Q_mx against each
    continuous variable, both pooled across models and broken out per
    model."""
    rows = []
    for model, group in [("ALL", df)] + list(df.groupby("model")):
        for var in CONTINUOUS_VARS:
            pearson_r, pearson_p = scipy_stats.pearsonr(group["Q_mx"], group[var])
            spearman_rho, spearman_p = scipy_stats.spearmanr(group["Q_mx"], group[var])
            rows.append(
                {
                    "model": model,
                    "variable": var,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "n": len(group),
                }
            )
    return pd.DataFrame(rows)


def categorical_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of Q_mx per bin of each categorical/binned variable, both
    pooled across models and broken out per model, plus eta-squared
    (between-group variance / total variance) as the categorical analogue of
    a correlation coefficient, and a one-way ANOVA F-test p-value for
    whether the bin means differ more than chance."""
    rows = []
    for model, group in [("ALL", df)] + list(df.groupby("model")):
        for var in CATEGORICAL_VARS:
            stats = group.groupby(var, observed=True)["Q_mx"].agg(
                ["mean", "std", "count"]
            )
            grand_mean = group["Q_mx"].mean()
            ss_between = ((stats["mean"] - grand_mean) ** 2 * stats["count"]).sum()
            ss_total = ((group["Q_mx"] - grand_mean) ** 2).sum()
            eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")

            samples = [
                sub["Q_mx"].to_numpy()
                for _, sub in group.groupby(var, observed=True)
                if len(sub) >= 2
            ]
            if len(samples) >= 2:
                _, anova_p = scipy_stats.f_oneway(*samples)
            else:
                anova_p = float("nan")

            for bin_label, row in stats.iterrows():
                rows.append(
                    {
                        "model": model,
                        "variable": var,
                        "bin": bin_label,
                        "mean": row["mean"],
                        "std": row["std"],
                        "n": int(row["count"]),
                        "eta_squared": eta_sq,
                        "anova_p": anova_p,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo_level", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: same directory as --repo_level",
    )
    args = parser.parse_args()

    repo_level = pd.read_csv(args.repo_level)
    strata = load_strata(args.dataset)
    df = repo_level.merge(strata, on="repo_id", how="inner")
    missing = set(repo_level["repo_id"]) - set(strata["repo_id"])
    if missing:
        print(
            f"[WARN] {len(missing)} repo_id(s) in --repo_level not found in --dataset, dropped"
        )

    cont = continuous_correlations(df)
    cat = categorical_breakdown(df)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\nCorrelation of Q(m,x) with continuous strata variables:")
    print(cont.to_string(index=False))

    print("\nQ(m,x) by categorical/binned strata variable (mean +/- std, eta^2):")
    print(cat.to_string(index=False))

    out_dir = args.output_dir or args.repo_level.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cont.to_csv(out_dir / "strata_correlation_continuous.csv", index=False)
    cat.to_csv(out_dir / "strata_correlation_categorical.csv", index=False)
    print(f"\nWrote strata correlation CSVs to {out_dir}")


if __name__ == "__main__":
    main()
