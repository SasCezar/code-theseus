"""Hierarchical Q(m,x,a) -> Q(m,x) -> Q(m) aggregation for judge scores.

Let M be the generator models, X the sampled repositories, A the artefact
types (req, intent, c4), and D the common evaluation dimensions shared by
every rubric: correctness, completeness, a task-specific 3rd criterion
(architectural_quality / requirements_quality / intent_quality, treated as
positionally common across tasks as "domain_quality"), consistency,
comprehensibility, conciseness and usefulness; see COMMON_DIMS_BY_VERSION.
Task-specific extras
(relationships, diagram_quality, per-section correctness) are diagnostics and
stay out of D.

The judge's holistic overall_score is also kept out of D: it restates the
seven dimensions, so averaging it in would count that judgement twice. It is
only compared with Q(m,x,a) as a check that the judge's summary agrees with
its own per-dimension scores.

For model m, repository x, artefact a, dimension d, r(m,x,a,d) is the judge's
rating. We compute:

    Q(m,x,a) = (1/|D|) * sum_d r(m,x,a,d)          artefact-level score
    Q(m,x)   = (1/|A|) * sum_a Q(m,x,a)            repository-level score
    Q(m)     = (1/|X|) * sum_x Q(m,x)              model-level score

Usage (several --scores files are repeated runs, averaged per document):
    uv run python -m reverse_se.evaluation.judge.aggregate \\
        --scores data/out/judge/judge_scores_1000stratified_v3_gpt-6-luna{,_run2,_run3}.jsonl \\
        --output_dir data/out/judge/aggregate_v3_gpt-6-luna_mean3
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from reverse_se.evaluation.judge.rubrics import RUBRIC_VERSION

DOMAIN_FIELD = {
    "c4": "architectural_quality",
    "req": "requirements_quality",
    "intent": "intent_quality",
}
COMMON_DIMS_BY_VERSION = {
    "v3": [
        "correctness",
        "completeness",
        "domain_quality",
        "consistency",
        "comprehensibility",
        "conciseness",
        "usefulness",
    ],
}
# Task-specific diagnostics, kept out of Q. Scores may be null (no
# diagrams, absent section), so each is reported with how many artefacts it
# was actually scored on.
EXTRA_SCORES = {
    "c4": [
        "relationships",
        "diagram_quality",
        "system_context_correctness",
        "container_level_correctness",
        "component_level_correctness",
    ],
    "req": ["fr_correctness", "nfr_correctness"],
}
SECTIONS = {
    "c4": ["system_context", "container_level", "component_level"],
    "req": ["fr", "nfr"],
}
IDENTIFIABILITY_LEVELS = ["clear", "partial", "absent"]


def load_rows(path: Path, rubric_version: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("rubric_version") != rubric_version:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rubric_version={rubric_version!r} rows found in {path}")
    df = pd.DataFrame(rows)
    # Keep the latest score for a given (model, task, repo_id) in case of a
    # resumed/overlapping run.
    return df.drop_duplicates(subset=["model", "task", "repo_id"], keep="last")


def load_runs(paths: list[Path], rubric_version: str) -> pd.DataFrame:
    """One row per (model, task, repo_id). With several score files (repeated
    judge runs), numeric columns are averaged per document across runs, so
    every score is a mean over runs; other columns (identifiability labels,
    justifications) are taken from the first run. n_runs says how many runs
    each document was scored in."""
    runs = [load_rows(path, rubric_version) for path in paths]
    if len(runs) == 1:
        return runs[0].assign(n_runs=1)
    key = ["model", "task", "repo_id"]
    stacked = pd.concat(runs, ignore_index=True)
    numeric = [
        c
        for c in stacked.columns
        if c not in key and pd.api.types.is_numeric_dtype(stacked[c])
    ]
    other = [c for c in stacked.columns if c not in key and c not in numeric]
    grouped = stacked.groupby(key, sort=False)
    df = pd.concat(
        [
            grouped[numeric].mean(),
            grouped[other].first(),
            grouped.size().rename("n_runs"),
        ],
        axis=1,
    ).reset_index()
    short = (df["n_runs"] < len(paths)).sum()
    print(
        f"Averaged {len(paths)} runs over {len(df)} documents"
        + (f" ({short} scored in fewer runs)" if short else ""),
        flush=True,
    )
    return df


def harmonize(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Add a `domain_quality` column mapping each row's task-specific 3rd
    criterion onto the common dimension name, and validate the rest of D
    is present."""
    df = df.copy()
    unknown_tasks = set(df["task"]) - set(DOMAIN_FIELD)
    if unknown_tasks:
        raise SystemExit(f"No domain-quality mapping for tasks: {unknown_tasks}")

    def domain_value(row: pd.Series) -> float:
        return row[DOMAIN_FIELD[row["task"]]]

    df["domain_quality"] = df.apply(domain_value, axis=1)

    missing = [d for d in dims if d != "domain_quality" and d not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected common-dimension columns: {missing}")
    return df


def compute_hierarchy(
    df: pd.DataFrame, dims: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (artefact_level_df, repo_level_df, model_level_df).

    model_level_df is tidy (model, mean, std), std computed across the |X|
    repo-level Q(m,x) values per model — i.e. how much a model's
    repository-level score varies across the 100 sampled repos."""
    artefact = df.copy()
    artefact["Q_mxa"] = artefact[dims].mean(axis=1)

    repo_level = (
        artefact.groupby(["model", "repo_id"])["Q_mxa"]
        .mean()
        .rename("Q_mx")
        .reset_index()
    )
    model_level = repo_level.groupby("model")["Q_mx"].agg(["mean", "std"]).reset_index()
    return artefact, repo_level, model_level


def per_dimension_model_scores(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """The same repo->model averaging hierarchy, kept separate per common
    dimension, for a diagnostic breakdown of *why* Q(m) differs across
    models (rather than just the single collapsed scalar). Tidy long format:
    one row per (model, dimension) with mean/std computed across repos."""
    rows = []
    for dim in dims:
        per_repo = df[["model", "repo_id", dim]].rename(columns={dim: "score"})
        repo_level = per_repo.groupby(["model", "repo_id"])["score"].mean()
        stats = repo_level.groupby("model").agg(["mean", "std"]).reset_index()
        stats["dimension"] = dim
        rows.append(stats)
    return pd.concat(rows, ignore_index=True)[["model", "dimension", "mean", "std"]]


def per_task_dimension_scores(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Per-dimension scores kept separate per artefact type too (unlike
    per_dimension_model_scores, which collapses across tasks). Tidy long
    format: one row per (task, model, dimension) with mean/std across the
    repos judged for that task -- no repo-level macro-averaging needed since
    each (model, repo, task) row is already a single artefact-level rating."""
    rows = []
    for task, task_df in df.groupby("task"):
        for dim in dims:
            stats = task_df.groupby("model")[dim].agg(["mean", "std"]).reset_index()
            stats["task"] = task
            stats["dimension"] = dim
            rows.append(stats)
    return pd.concat(rows, ignore_index=True)[
        ["task", "model", "dimension", "mean", "std"]
    ]


def extra_score_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of each task-specific score per (task, model), over the
    artefacts where it was scored. n_scored < n_artefacts means the judge
    returned null, e.g. pct_scored for diagram_quality is the share of C4
    documents that contain diagrams at all."""
    rows = []
    for task, metrics in EXTRA_SCORES.items():
        task_df = df[df["task"] == task]
        for metric in metrics:
            if metric not in task_df.columns:
                continue
            scores = pd.to_numeric(task_df[metric], errors="coerce")
            stats = scores.groupby(task_df["model"]).agg(["mean", "std", "count"])
            stats["n_artefacts"] = task_df.groupby("model").size()
            stats = stats.rename(columns={"count": "n_scored"}).reset_index()
            stats["pct_scored"] = (
                100 * stats["n_scored"] / stats["n_artefacts"]
            ).round(1)
            stats["task"] = task
            stats["metric"] = metric
            rows.append(stats)
    columns = [
        "task",
        "model",
        "metric",
        "mean",
        "std",
        "n_scored",
        "n_artefacts",
        "pct_scored",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def identifiability_stats(df: pd.DataFrame) -> pd.DataFrame:
    """% of artefacts per (task, model, section) at each identifiability
    level (clear / partial / absent)."""
    rows = []
    for task, sections in SECTIONS.items():
        task_df = df[df["task"] == task]
        for section in sections:
            column = f"{section}_identifiability"
            if column not in task_df.columns:
                continue
            pct = (
                pd.crosstab(task_df["model"], task_df[column], normalize="index")
                .reindex(columns=IDENTIFIABILITY_LEVELS, fill_value=0.0)
                .mul(100)
                .round(1)
                .reset_index()
            )
            pct["task"] = task
            pct["section"] = section
            rows.append(pct)
    columns = ["task", "model", "section", *IDENTIFIABILITY_LEVELS]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def overall_coherence(artefact: pd.DataFrame) -> pd.DataFrame:
    """Agreement between the judge's holistic overall score and the computed
    Q(m,x,a), per (task, model), with ALL rows pooling across tasks and/or
    models. A high Spearman rho means the judge's summary matches its own
    per-dimension scores; mean_diff > 0 means the summary is more lenient."""
    groups = [
        (task, model, sub)
        for task, task_df in [("ALL", artefact), *artefact.groupby("task")]
        for model, sub in [("ALL", task_df), *task_df.groupby("model")]
    ]
    rows = []
    for task, model, sub in groups:
        overall = pd.to_numeric(sub["overall"], errors="coerce")
        rows.append(
            {
                "task": task,
                "model": model,
                "mean_overall": overall.mean(),
                "mean_Q": sub["Q_mxa"].mean(),
                "mean_diff": (overall - sub["Q_mxa"]).mean(),
                "spearman_rho": overall.corr(sub["Q_mxa"], method="spearman"),
                "n": int(overall.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def per_artefact_model_scores(artefact: pd.DataFrame) -> pd.DataFrame:
    """Q(m,x,a) averaged over repos only (not macro-averaged across
    artefacts), i.e. a per-task breakdown of each model's performance. Tidy
    long format: one row per (model, task) with mean/std across repos."""
    return (
        artefact.groupby(["model", "task"])["Q_mxa"].agg(["mean", "std"]).reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scores",
        type=Path,
        nargs="+",
        required=True,
        help="One or more score files; several (repeated runs) are averaged per document",
    )
    parser.add_argument(
        "--rubric_version",
        choices=list(COMMON_DIMS_BY_VERSION),
        default=RUBRIC_VERSION,
        help=f"Which rubric version's rows to aggregate (default: {RUBRIC_VERSION})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: same directory as the first --scores file",
    )
    args = parser.parse_args()

    dims = COMMON_DIMS_BY_VERSION[args.rubric_version]
    df = load_runs(args.scores, args.rubric_version)
    df = harmonize(df, dims)
    artefact, repo_level, model_level = compute_hierarchy(df, dims)
    per_dim = per_dimension_model_scores(df, dims)
    by_task = per_artefact_model_scores(artefact)
    by_task_dim = per_task_dimension_scores(df, dims)

    ranked = model_level.sort_values("mean", ascending=False)["model"]

    def display(tidy: pd.DataFrame, index_col: str) -> str:
        pivoted = tidy.pivot(index="model", columns=index_col)
        cells = (
            pivoted["mean"].round(3).astype(str)
            + " ± "
            + pivoted["std"].round(3).astype(str)
        )
        return cells.loc[ranked].to_string()

    print("\nQ(m) - model-level score, mean +/- std across the sampled repos:")
    print(
        (
            model_level.set_index("model").loc[ranked, "mean"].round(3).astype(str)
            + " ± "
            + model_level.set_index("model").loc[ranked, "std"].round(3).astype(str)
        ).to_string()
    )

    print("\nPer-dimension Q(m) breakdown (mean +/- std across repos):")
    print(display(per_dim, "dimension"))

    print(
        "\nQ(m,x,a) averaged over repos only - per-artefact-type breakdown (mean +/- std):"
    )
    print(display(by_task, "task"))

    print("\nPer-(task, dimension) breakdown (mean across repos):")
    for task in sorted(by_task_dim["task"].unique()):
        print(f"\n  task={task}")
        pivoted = (
            by_task_dim[by_task_dim["task"] == task]
            .pivot(index="model", columns="dimension", values="mean")
            .round(3)
        )
        print(pivoted.loc[[m for m in ranked if m in pivoted.index]].to_string())

    extras = extra_score_stats(df)
    identifiability = identifiability_stats(df)
    if not extras.empty:
        print("\nTask-specific scores, not part of Q (mean over scored artefacts):")
        for task in sorted(extras["task"].unique()):
            print(f"\n  task={task}")
            task_extras = extras[extras["task"] == task]
            means = task_extras.pivot(index="model", columns="metric", values="mean")
            pct = task_extras.pivot(
                index="model", columns="metric", values="pct_scored"
            )
            cells = means.round(3).astype(str) + " (" + pct.astype(str) + "%)"
            print(cells.loc[[m for m in ranked if m in cells.index]].to_string())
        print("  (n%) = share of artefacts the judge scored; null otherwise")

        print("\nSection identifiability, % of artefacts (clear / partial / absent):")
        for task in sorted(identifiability["task"].unique()):
            print(f"\n  task={task}")
            print(
                identifiability[identifiability["task"] == task]
                .drop(columns="task")
                .to_string(index=False)
            )

    coherence = overall_coherence(artefact)
    print(
        "\nJudge's holistic overall vs computed Q(m,x,a), not part of Q "
        "(mean_diff = overall - Q):"
    )
    print(coherence.round(3).to_string(index=False))

    out_dir = args.output_dir or args.scores[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    artefact.drop(columns=["context_budget"]).to_csv(
        out_dir / "Q_artefact_level.csv", index=False
    )
    repo_level.to_csv(out_dir / "Q_repo_level.csv", index=False)
    model_level.to_csv(out_dir / "Q_model_level.csv", index=False)
    per_dim.to_csv(out_dir / "Q_model_by_dimension.csv", index=False)
    by_task.rename(columns={"task": "artefact"}).to_csv(
        out_dir / "Q_model_by_artefact.csv", index=False
    )
    by_task_dim.rename(columns={"task": "artefact"}).to_csv(
        out_dir / "Q_model_by_artefact_dimension.csv", index=False
    )
    coherence.to_csv(out_dir / "overall_coherence.csv", index=False)
    if not extras.empty:
        extras.to_csv(out_dir / "extra_scores.csv", index=False)
        identifiability.to_csv(out_dir / "section_identifiability.csv", index=False)
    print(f"\nWrote judge aggregate CSVs to {out_dir}")


if __name__ == "__main__":
    main()
