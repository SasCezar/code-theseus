#!/usr/bin/env python3
"""Analyze Commit0 benchmark results: intent_only vs req_arch.

Reads <out>/results.jsonl (benchmark_commit0.py), keeps valid runs, and
prints:
  - per-library table, replicates averaged per condition: held-out /
    visible pass rate, output and input tokens, task tokens (the treatment:
    intent vs intent + requirements + architecture), wall time, steps;
  - paired comparison over libraries with both conditions (req_arch -
    intent_only): wins, mean / median difference, bootstrap 95% CI of the
    mean, Wilcoxon signed-rank and sign tests;
  - efficiency: held-out pass rate (pp) per 100k output tokens, per run;
  - with --trajectories: steps until the agent's own test runs first passed
    50% / 80% of the visible tests (a measure of speed that doesn't depend
    on runs hitting the step cap).

Invalid runs are listed and excluded: audit failures (the agent saw another
copy of the library), interrupted launches, and grading errors.

Usage:
    uv run scripts/python/analyze_commit0.py [--out data/out/benchmark/commit0]
        [--scaffold contract] [--model hosted_vllm/qwen3.8-flash-next]
        [--trajectories] [--csv per_library.csv] [--exclude chardet ...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = ("intent_only", "req_arch")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_PYTEST_COUNT = re.compile(r"(\d+) passed")


def load(args) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in (args.out / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(rows)
    if args.scaffold:
        df = df[df["scaffold"] == args.scaffold]
    if args.model:
        df = df[df["model"] == args.model]
    if args.exclude:
        df = df[~df["name"].isin(args.exclude)]
    return df


def split_valid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reasons = pd.Series("", index=df.index)
    reasons[~df["audit_ok"].fillna(False)] += "audit "
    reasons[df["terminal_status"] == "launch_error"] += "launch_error "
    reasons[df["agent_steps"].fillna(0) == 0] += "no_steps "
    reasons[df["grade_error"].notna()] += "grade_error "
    bad = df[reasons != ""].assign(reason=reasons[reasons != ""].str.strip())
    return df[reasons == ""], bad


def steps_to_threshold(run_dir: Path, visible_total: int, share: float) -> int | None:
    """First agent step at which one of its own pytest runs reported at
    least share * visible_total passed tests (None if never)."""
    trajectory = run_dir / "trajectory.jsonl"
    if not trajectory.is_file() or not visible_total:
        return None
    step = 0
    for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") == "ActionEvent":
            step += 1
        elif event.get("kind") == "ObservationEvent":
            content = (event.get("observation") or {}).get("content") or []
            text = _ANSI.sub(
                "", "".join(c.get("text", "") for c in content if isinstance(c, dict))
            )
            counts = [int(n) for n in _PYTEST_COUNT.findall(text)]
            if counts and max(counts) >= share * visible_total:
                return step
    return None


def per_library(valid: pd.DataFrame, args) -> pd.DataFrame:
    metrics = {
        "held_out_pass_rate": "mean",
        "visible_pass_rate": "mean",
        "output_tokens": "mean",
        "input_tokens": "mean",
        "task_tokens": "mean",
        "wall_time_seconds": "mean",
        "agent_steps": "mean",
        "held_out_passed": "mean",
        "run_id": "count",
    }
    frame = valid.copy()
    if "task_tokens" not in frame:
        frame["task_tokens"] = np.nan
    if args.trajectories:
        runs = args.out / "runs"
        for share in (0.5, 0.8):
            frame[f"steps_to_{int(share * 100)}"] = [
                steps_to_threshold(
                    runs
                    / r.model.replace("/", "__")
                    / r.scaffold
                    / r.name
                    / r.condition
                    / f"{int(r.replicate):02d}",
                    int(r.visible_total or 0),
                    share,
                )
                for r in frame.itertuples()
            ]
            metrics[f"steps_to_{int(share * 100)}"] = "mean"
    grouped = frame.groupby(["name", "condition"]).agg(metrics)
    grouped = grouped.rename(columns={"run_id": "n"})
    return grouped.unstack("condition")


def paired(table: pd.DataFrame) -> pd.DataFrame:
    rate = table["held_out_pass_rate"]
    both = rate.dropna(subset=list(CONDITIONS))
    out = pd.DataFrame(
        {
            "intent_only": both["intent_only"],
            "req_arch": both["req_arch"],
            "diff_pp": 100 * (both["req_arch"] - both["intent_only"]),
        }
    )
    tokens = table["output_tokens"].loc[both.index]
    out["output_tokens_diff_pct"] = 100 * (
        tokens["req_arch"] / tokens["intent_only"] - 1
    )
    return out


def summarize(pairs: pd.DataFrame, seed: int = 0) -> dict:
    diffs = pairs["diff_pp"].to_numpy()
    n = len(diffs)
    result = {"libraries": n}
    if n == 0:
        return result
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(10_000, n), replace=True).mean(axis=1)
    result.update(
        req_arch_better=int((diffs > 0).sum()),
        intent_better=int((diffs < 0).sum()),
        mean_diff_pp=float(diffs.mean()),
        median_diff_pp=float(np.median(diffs)),
        mean_diff_ci95=(
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ),
    )
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 1:
        result["wilcoxon_p"] = float(stats.wilcoxon(nonzero).pvalue)
        result["sign_test_p"] = float(
            stats.binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/out/benchmark/commit0"))
    parser.add_argument("--scaffold", default="contract")
    parser.add_argument("--model", default=None)
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Libraries to leave out, e.g. chardet (its held-out tests' data "
        "files are in the workspace)",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Write per-library table"
    )
    args = parser.parse_args()

    df = load(args)
    valid, bad = split_valid(df)
    print(f"{len(df)} runs, {len(valid)} valid, {len(bad)} excluded")
    if len(bad):
        print(bad[["name", "condition", "replicate", "reason"]].to_string(index=False))

    table = per_library(valid, args)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print("\nPer library (replicates averaged):")
    print(table.round(3).to_string())
    if args.csv:
        table.to_csv(args.csv)

    pairs = paired(table)
    print("\nPaired held-out pass rate (req_arch - intent_only):")
    print(pairs.round(3).to_string())
    print("\nSummary:")
    for key, value in summarize(pairs).items():
        print(f"  {key}: {value}")

    # Per run, then averaged: pooling tests would let babel (5,663) dominate.
    eff = (
        valid.assign(
            eff=100 * valid["held_out_pass_rate"] / (valid["output_tokens"] / 1e5)
        )
        .groupby("condition")["eff"]
        .mean()
    )
    print("\nHeld-out pass rate (pp) per 100k output tokens, mean over runs:")
    print(eff.round(2).to_string())
    if len(valid.groupby(["name", "condition"]).filter(lambda g: len(g) > 1)):
        sd = valid.groupby(["name", "condition"])["held_out_pass_rate"].std().dropna()
        print(
            f"\nWithin-condition SD of held-out rate across replicates: "
            f"mean {100 * sd.mean():.1f} pp over {len(sd)} (library, condition) cells"
        )


if __name__ == "__main__":
    main()
