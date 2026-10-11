#!/usr/bin/env python3
"""Judge Q of the Commit0 artefacts vs the agent's gain (Full - Intent).

Usage:
    uv run --with statsmodels scripts/python/judge_gain_commit0.py \\
        data/out/judge/judge_scores_commit0_v3_gpt-6-luna.jsonl \\
        data/out/benchmark/agent_libraries_generators.csv
"""

import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from reverse_se.evaluation.judge.aggregate import (
    COMMON_DIMS_BY_VERSION,
    harmonize,
    load_rows,
)

scores, gains_csv = Path(sys.argv[1]), Path(sys.argv[2])
dims = COMMON_DIMS_BY_VERSION["v3"]
df = harmonize(load_rows(scores, "v3"), dims)
df["Q"] = df[dims].mean(axis=1)
df["generator"] = df["model"].map(
    {"hosted_vllm--gemma-4-31b": "Gemma", "hosted_vllm--qwen-3.5-122b": "Q35"}
)
df["name"] = df["repo_id"].str.split("|").str[1]
wide = df.pivot_table(index=["name", "generator"], columns="task", values="Q")
corr = df.pivot_table(index=["name", "generator"], columns="task", values="correctness")
pred = pd.DataFrame(
    {
        "Q": wide.mean(axis=1),  # Q(m,x), all three artefacts
        "Q_reqarch": wide[["req", "c4"]].mean(axis=1),  # what Full adds
        "corr_reqarch": corr[["req", "c4"]].mean(axis=1),
    }
).reset_index()
print(pred.groupby("generator")[["Q", "Q_reqarch"]].mean().round(2), "\n")

gains = pd.read_csv(gains_csv)
gains = gains[gains["generator"] != "Pooled"]
for outcome in ["module_f1", "held_out_pass_rate"]:
    d = gains[gains["outcome"] == outcome].merge(pred, on=["name", "generator"])
    print(f"== {outcome}: {len(d)} units, {d['name'].nunique()} libraries")
    for p in ["Q", "Q_reqarch", "corr_reqarch"]:
        rho, pv = spearmanr(d[p], d["diff"])
        line = f"  {p:13s} pooled rho={rho:+.2f} p={pv:.3f}"
        for g, sub in d.groupby("generator"):
            r, q = spearmanr(sub[p], sub["diff"])
            line += f" | {g} rho={r:+.2f} p={q:.3f}"
        z = d.assign(x=(d[p] - d[p].mean()) / d[p].std())
        m = smf.mixedlm("diff ~ x", z, groups=z["name"]).fit(reml=True)
        line += f" | mixed beta={m.params['x']:+.3f} p={m.pvalues['x']:.3f}"
        print(line)
