"""Human-readable score summary + tidy long-format CSV export for judge_scores.jsonl."""

import csv
import json
from collections import defaultdict
from pathlib import Path

SCORE_EXCLUDE = {
    "comments",
    "_usage",
    "model",
    "context_budget",
    "task",
    "repo_id",
    "judge_model",
    "rubric_version",
}


def _read_rows(output_path: Path) -> list[dict]:
    rows = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _numeric_items(row: dict) -> list[tuple[str, float]]:
    return [
        (k, v)
        for k, v in row.items()
        if k not in SCORE_EXCLUDE and isinstance(v, (int, float))
    ]


def build_summary(output_path: Path) -> str | None:
    """Build the human-readable judge score summary table as a string, or
    None if there are no scored rows to summarize."""
    rows = _read_rows(output_path)
    if not rows:
        return None

    # Aggregate numeric scores per model
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    row_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        model = row["model"]
        row_counts[model] += 1
        for k, v in _numeric_items(row):
            sums[model][k] += v
            counts[model][k] += 1

    # Per-model mean scores; rank by "overall" if present else grand mean
    summary: dict[str, tuple[float, dict[str, float]]] = {}
    all_keys = sorted({k for m in sums for k in sums[m]})
    for model in sums:
        means = {k: sums[model][k] / counts[model][k] for k in sums[model]}
        rank_score = means.get(
            "overall", sum(means.values()) / len(means) if means else 0
        )
        summary[model] = (rank_score, means)

    # Group metrics by shared prefix (context_, container_, component_, fr_, nfr_)
    prefix_map: dict[str, list[str]] = defaultdict(list)
    for k in all_keys:
        if k == "overall":
            continue
        parts = k.rsplit("_", 1)
        prefix_map[parts[0] if len(parts) == 2 else k].append(k)
    groups: list[tuple[str, list[str]]] = sorted(prefix_map.items())
    groups.append(("overall", ["overall"]))

    def group_avg(means: dict, keys: list[str]) -> float:
        vals = [means[k] for k in keys if k in means]
        return sum(vals) / len(vals) if vals else float("nan")

    col_w = max(max(len(m) for m in summary), 5) + 2
    metric_w = 20
    models_ranked = sorted(summary, key=lambda m: summary[m][0], reverse=True)

    lines = ["\n--- Judge Score Summary (sorted by overall) ---"]
    lines.append(
        f"{'n':<{metric_w}}"
        + "".join(f"{row_counts[m]:>{col_w}}" for m in models_ranked)
    )
    header = f"{'metric':<{metric_w}}" + "".join(f"{m:>{col_w}}" for m in models_ranked)
    lines.append(header)
    lines.append("-" * len(header))

    for g_name, keys in groups:
        if len(keys) > 1:
            avg_label = f"{g_name} (avg)"
            lines.append(
                f"{avg_label:<{metric_w}}"
                + "".join(
                    f"{group_avg(summary[m][1], keys):>{col_w}.2f}"
                    for m in models_ranked
                )
            )
            for k in keys:
                label = f"  {k.split('_')[-1]}"
                lines.append(
                    f"{label:<{metric_w}}"
                    + "".join(
                        f"{summary[m][1].get(k, float('nan')):>{col_w}.2f}"
                        for m in models_ranked
                    )
                )
        else:
            k = keys[0]
            lines.append(
                f"{k:<{metric_w}}"
                + "".join(
                    f"{summary[m][1].get(k, float('nan')):>{col_w}.2f}"
                    for m in models_ranked
                )
            )
    lines.append("")
    return "\n".join(lines)


def print_summary(output_path: Path, summary_path: Path | None = None) -> None:
    """Print the judge score summary, and optionally persist it to
    summary_path so it's a durable artifact of the results generation run."""
    text = build_summary(output_path)
    if text is None:
        return
    print(text)
    if summary_path is not None:
        summary_path.write_text(text.strip() + "\n", encoding="utf-8")
        print(f"Summary: {summary_path}")


def write_csv(output_path: Path, csv_path: Path) -> None:
    rows = _read_rows(output_path)
    if not rows:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "context_budget",
                "task",
                "repo_id",
                "judge_model",
                "rubric_version",
                "metric",
                "score",
            ]
        )
        for row in rows:
            meta = [
                row.get("model", ""),
                row.get("context_budget", ""),
                row.get("task", ""),
                row.get("repo_id", ""),
                row.get("judge_model", ""),
                row.get("rubric_version", ""),
            ]
            for k, v in _numeric_items(row):
                writer.writerow([*meta, k, v])
