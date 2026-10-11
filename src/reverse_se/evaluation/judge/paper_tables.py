"""LaTeX tables and \\newcommand macros for the paper's LLM-as-a-judge
results, from one or more judge score files (repeated runs of the same judge
and rubric, averaged per document as in judge/aggregate.py).

Writes to --tex_dir:
  judge_main.tex         Q per generator (bootstrap 95% CI), per artefact type
                         and per common dimension (mean over runs)
  judge_reliability.tex  run-to-run reliability per dimension (needs >= 2 runs)
  judge_strata.tex       Q of every generator per sampling stratum (repository
                         size and popularity), with --dataset
  judge_numbers.tex      macros for the numbers quoted in the text

Usage:
    uv run python -m reverse_se.evaluation.judge.paper_tables \\
        --scores data/out/judge/judge_scores_1000stratified_v3_gpt-6-luna{,_run2,_run3}.jsonl \\
        --labels hosted_vllm--gemma-4-31b='\\mGemma=Gemma' \\
                 hosted_vllm--qwen-3.5-122b='\\mQwen=Qwen' \\
                 hosted_vllm--gpt-oss-120b='\\mGpto=Gpto' \\
        --dataset data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000_stratified.jsonl \\
        --tex_dir paper/tables
"""

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from reverse_se.evaluation.judge.aggregate import (
    COMMON_DIMS_BY_VERSION,
    DOMAIN_FIELD,
    harmonize,
    load_rows,
    load_runs,
)
from reverse_se.evaluation.judge.rubrics import RUBRIC_VERSION
from reverse_se.evaluation.judge.strata_correlation import (
    SIZE_BIN_LABELS,
    STAR_BIN_LABELS,
    load_strata,
)

KEY = ["model", "task", "repo_id"]
TASK_NAMES = {"intent": "Intent", "req": "Requirements", "c4": "Architecture"}
TASK_SHORT = {"intent": "Intent", "req": "Req.", "c4": "Arch."}
DIM_NAMES = {
    "correctness": ("Correctness", "Corr."),
    "completeness": ("Completeness", "Compl."),
    "domain_quality": ("Abstraction", "Abstr."),
    "consistency": ("Consistency", "Cons."),
    "comprehensibility": ("Clarity", "Clar."),
    "conciseness": ("Conciseness", "Conc."),
    "usefulness": ("Usefulness", "Use."),
}
# Artefact-specific ratings, outside Q: column -> (artefact, rating fields
# averaged into it)
SPECIFIC = {
    "Rel.": ("c4", ["relationships"]),
    "Diag.": ("c4", ["diagram_quality"]),
    "C4": (
        "c4",
        [
            "system_context_correctness",
            "container_level_correctness",
            "component_level_correctness",
        ],
    ),
    "FR": ("req", ["fr_correctness"]),
    "NFR": ("req", ["nfr_correctness"]),
}
BOOTSTRAP_SAMPLES = 2000


def signed(x: float) -> str:
    """Two-decimal value with a sign, without printing -0.00."""
    return f"{round(x, 2) + 0.0:+.2f}"


def cmd(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"


def fmt_p(p: float) -> str:
    """p-value for the text: plain to three decimals, or as a power of ten
    when tiny (e.g. $<10^{-100}$)."""
    if p >= 0.001:
        return f"{p:.3f}"
    return f"$<10^{{{math.ceil(math.log10(p))}}}$"


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple:
    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def icc_consistency(mat: np.ndarray) -> tuple[float, float]:
    """Two-way consistency ICC over documents (rows) x runs (columns):
    (ICC(3,1) for a single run, ICC(3,k) for the mean of the k runs)."""
    n, k = mat.shape
    grand = mat.mean()
    ss_rows = k * ((mat.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((mat.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((mat - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    return (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err), (
        ms_rows - ms_err
    ) / ms_rows


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def with_q(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    df = harmonize(df, dims)
    df[dims] = df[dims].apply(pd.to_numeric, errors="coerce")
    df["Q"] = df[dims].mean(axis=1)
    return df


def main_table(df, dims, labels, macros, rng) -> list[str]:
    repo = df.groupby(["model", "repo_id"])["Q"].mean().unstack("model")
    by_task = df.groupby(["model", "task"])["Q"].mean().unstack("task")
    by_dim = df.groupby("model")[dims].mean()
    specific = {}
    for col, (task, fields) in SPECIFIC.items():
        own = df[df["task"] == task]
        rating = own[fields].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        specific[col] = rating.groupby(own["model"]).mean()
    values = pd.DataFrame(
        {
            "Q": repo.mean(),
            **{t: by_task[t] for t in TASK_NAMES},
            **{d: by_dim[d] for d in dims},
            **specific,
        }
    ).loc[list(labels)]
    best = values.max()
    # Spearman of the holistic rating with Q(m,x,a), per generator
    holistic = {
        model: stats.spearmanr(
            pd.to_numeric(own["overall"], errors="coerce"), own["Q"], nan_policy="omit"
        )[0]
        for model, own in df.groupby("model")
    }

    def cell(model, col):
        text = f"{values.loc[model, col]:.2f}"
        return f"\\textbf{{{text}}}" if values.loc[model, col] == best[col] else text

    lines = [
        "\\begin{table*}[htbp!]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4.2pt}",
        "\\caption{Judge scores (1--5) per generator, averaged over runs. $Q$:"
        " mean of $Q(m,x)$ over repositories; per artefact: mean of $Q(m,x,a)$;"
        " per dimension and artefact-specific: mean rating (abbreviations as in"
        " Table~\\ref{tab:quality_dimensions}; C4: mean over the three views)."
        " Best in bold.}",
        "\\label{tab:judge_main}",
        "\\begin{tabular}{l"
        + "c"
        + "ccc"
        + "c" * len(dims)
        + "c" * len(SPECIFIC)
        + "}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{\\textbf{Per artefact}}"
        f" & \\multicolumn{{{len(dims)}}}{{c}}{{\\textbf{{Per dimension}}}}"
        f" & \\multicolumn{{{len(SPECIFIC)}}}{{c}}{{\\textbf{{Artefact-specific}}}} \\\\",
        f"\\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-{5 + len(dims)}}}"
        f" \\cmidrule(lr){{{6 + len(dims)}-{5 + len(dims) + len(SPECIFIC)}}}",
        "\\textbf{Model} & \\textbf{$Q$} & "
        + " & ".join(f"\\textbf{{{TASK_SHORT[t]}}}" for t in TASK_NAMES)
        + " & "
        + " & ".join(f"\\textbf{{{DIM_NAMES[d][1]}}}" for d in dims)
        + " & "
        + " & ".join(f"\\textbf{{{col}}}" for col in SPECIFIC)
        + " \\\\",
        "\\midrule",
    ]
    for model, (label, prefix) in labels.items():
        low, high = bootstrap_ci(repo[model].dropna().to_numpy(), rng)
        q = cell(model, "Q")
        cells = [label, q]
        cells += [cell(model, t) for t in TASK_NAMES]
        cells += [cell(model, d) for d in dims]
        cells += [cell(model, col) for col in SPECIFIC]
        lines.append(" & ".join(cells) + " \\\\")
        macros += [
            cmd(f"{prefix}HolisticRho", f"{holistic[model]:.2f}"),
            cmd(f"{prefix}Q", f"{values.loc[model, 'Q']:.2f}"),
            cmd(f"{prefix}QLow", f"{low:.2f}"),
            cmd(f"{prefix}QHigh", f"{high:.2f}"),
            *(
                cmd(f"{prefix}{TASK_NAMES[t]}Q", f"{values.loc[model, t]:.2f}")
                for t in TASK_NAMES
            ),
            *(
                cmd(f"{prefix}{DIM_NAMES[d][0]}", f"{values.loc[model, d]:.2f}")
                for d in dims
            ),
        ]
    macros.append(
        cmd("HolisticRhoMean", f"{np.mean([holistic[m] for m in labels]):.2f}")
    )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]

    pairs = list(itertools.combinations(labels, 2))
    pvalues = []
    for a, b in pairs:
        both = repo[[a, b]].dropna()
        pvalues.append(stats.wilcoxon(both[a], both[b]).pvalue)
    for (a, b), p_holm in zip(pairs, holm(pvalues), strict=True):
        both = repo[[a, b]].dropna()
        diff = both[a] - both[b]
        name = f"{labels[a][1]}Vs{labels[b][1]}"
        macros += [
            cmd(f"{name}Diff", f"{diff.mean():+.2f}"),
            cmd(f"{name}Higher", f"{100 * (diff > 0).mean():.0f}"),
            cmd(f"{name}Holm", fmt_p(p_holm)),
        ]
    return lines


def reliability_table(run_frames, dims, macros) -> list[str]:
    common = run_frames[0].set_index(KEY).index
    for frame in run_frames[1:]:
        common = common.intersection(frame.set_index(KEY).index)
    runs = [f.set_index(KEY).loc[common] for f in run_frames]
    k = len(runs)
    rows = []
    for col in [*dims, "Q", "overall"]:
        mat = np.column_stack([r[col].to_numpy(dtype=float) for r in runs])
        rho = np.mean(
            [
                stats.spearmanr(mat[:, i], mat[:, j])[0]
                for i, j in itertools.combinations(range(k), 2)
            ]
        )
        single, mean_k = icc_consistency(mat)
        rows.append((col, rho, single, mean_k))
    name = {**{d: DIM_NAMES[d][0] for d in dims}, "Q": "$Q$", "overall": "Holistic"}
    lines = [
        "\\begin{table}[htbp!]",
        "\\centering",
        "\\small",
        f"\\caption{{Agreement of the judge with itself across {k} independent"
        f" runs over the same {len(common):,} artefacts: mean pairwise Spearman"
        " $\\rho$ between runs, and the two-way consistency intraclass"
        " correlation for a single run, ICC(3,1), and for the mean of the runs,"
        f" ICC(3,{k}).}}",
        "\\label{tab:judge_reliability}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Dimension} & \\textbf{$\\rho$} & \\textbf{ICC, one run}"
        f" & \\textbf{{ICC, mean of {k}}} \\\\",
        "\\midrule",
    ]
    for col, rho, single, mean_k in rows:
        if col == "Q":
            lines.append("\\midrule")
        lines.append(f"{name[col]} & {rho:.2f} & {single:.2f} & {mean_k:.2f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    by = {col: (rho, single, mean_k) for col, rho, single, mean_k in rows}
    dim_single = [by[d][1] for d in dims]
    dim_mean = [by[d][2] for d in dims]
    repo_runs = [r.groupby(level=["model", "repo_id"])["Q"].mean() for r in runs]
    repo_rho = np.mean(
        [
            stats.spearmanr(repo_runs[i], repo_runs[j])[0]
            for i, j in itertools.combinations(range(k), 2)
        ]
    )
    macros += [
        cmd("JudgeRuns", k),
        cmd("QIccSingle", f"{by['Q'][1]:.2f}"),
        cmd("QIccMean", f"{by['Q'][2]:.2f}"),
        cmd("QRunRho", f"{by['Q'][0]:.2f}"),
        cmd("RepoRunRho", f"{repo_rho:.2f}"),
        cmd("DimIccSingleMin", f"{min(dim_single):.2f}"),
        cmd("DimIccSingleMax", f"{max(dim_single):.2f}"),
        cmd("DimIccMeanMin", f"{min(dim_mean):.2f}"),
        cmd("DimIccMeanMax", f"{max(dim_mean):.2f}"),
        cmd("ClarityIccSingle", f"{by['comprehensibility'][1]:.2f}"),
        cmd("ClarityIccMean", f"{by['comprehensibility'][2]:.2f}"),
    ]
    # Largest gap in Q(m) between any two runs, to show the ranking is stable.
    per_run_q = [r.groupby(level=["model", "repo_id"])["Q"].mean() for r in runs]
    per_run_model = pd.concat(
        [q.groupby(level="model").mean() for q in per_run_q], axis=1
    )
    macros.append(
        cmd(
            "QRunSpread",
            f"{(per_run_model.max(axis=1) - per_run_model.min(axis=1)).max():.3f}",
        )
    )
    return lines


def diagnostics(df, dims, labels, macros) -> None:
    extras = {
        "Relationships": ("c4", "relationships"),
        "Diagram": ("c4", "diagram_quality"),
        "Fr": ("req", "fr_correctness"),
        "Nfr": ("req", "nfr_correctness"),
    }
    for model, (_label, prefix) in labels.items():
        own = df[df["model"] == model]
        for name, (task, col) in extras.items():
            vals = pd.to_numeric(own[own["task"] == task][col], errors="coerce")
            macros.append(cmd(f"{prefix}{name}", f"{vals.mean():.2f}"))
    overall = pd.to_numeric(df["overall"], errors="coerce")
    macros += [
        cmd("HolisticRho", f"{stats.spearmanr(overall, df['Q'])[0]:.2f}"),
        cmd("HolisticDiff", f"{(overall - df['Q']).mean():+.2f}"),
        cmd("JudgeDocs", f"{len(df):,}"),
    ]


STRATA = {
    "size_bin": (
        "Size",
        "num_nodes",
        SIZE_BIN_LABELS,
        ["$<$100", "100--1k", "$>$1k"],
    ),
    "star_bin": (
        "Popularity",
        "stars",
        STAR_BIN_LABELS,
        ["$<$500", "500--5k", "$>$5k"],
    ),
}


def strata_table(df, strata, labels, macros) -> list[str]:
    """Q of every generator per sampling stratum: repository size (code-graph
    nodes) and popularity (stars), overall and per artefact type, with the
    Spearman correlation of Q with the underlying count below each stratum."""
    merged = df.merge(strata, on="repo_id")
    n_cols = 1 + len(TASK_NAMES)
    lines = [
        "\\begin{table*}[htbp!]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\caption{Quality score $Q$ per generator and sampling stratum: repository"
        " size (nodes of the code graph) and popularity (stars), overall and per"
        " artefact type (Intent, Requirements, Architecture). The $\\rho$ rows give"
        " the Spearman correlation of the score with the number of nodes or stars"
        " over all repositories ($^{*}$: $p < 0.05$).}",
        "\\label{tab:judge_strata}",
        "\\begin{tabular}{@{}llr" + "c" * (n_cols * len(labels)) + "@{}}",
        "\\toprule",
        " & & "
        + "".join(
            f" & \\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{label}}}}}"
            for label, _ in labels.values()
        )
        + " \\\\",
        " ".join(
            f"\\cmidrule(lr){{{4 + i * n_cols}-{3 + (i + 1) * n_cols}}}"
            for i in range(len(labels))
        ),
        "\\textbf{Stratum} & & \\textbf{Repos}"
        + (
            " & \\textbf{$Q$}"
            + "".join(f" & \\textbf{{{TASK_SHORT[t]}}}" for t in TASK_NAMES)
        )
        * len(labels)
        + " \\\\",
        "\\midrule",
    ]

    def rho_cell(x, y):
        rho, p = stats.spearmanr(x, y)
        return rho, p, f"${signed(rho)}" + ("^{*}" if p < 0.05 else "") + "$"

    for col, (title, count, bins, ranges) in STRATA.items():
        # rho per generator, overall ("") and per artefact type, for the
        # cross-generator ranges quoted in the text
        rhos = {key: [] for key in ["", *TASK_NAMES.values()]}
        for i, (b, rng_text) in enumerate(zip(bins, ranges, strict=True)):
            in_bin = merged[merged[col] == b]
            n_repos = {
                in_bin[in_bin["model"] == m]["repo_id"].nunique() for m in labels
            }
            assert len(n_repos) == 1, f"generators cover different repos in {b}"
            cells = [
                f"\\multirow{{3}}{{*}}{{{title}}}" if i == 0 else "",
                f"{b} ({rng_text})",
                f"{n_repos.pop():,}",
            ]
            for model in labels:
                own = in_bin[in_bin["model"] == model]
                cells.append(f"{own.groupby('repo_id')['Q'].mean().mean():.2f}")
                cells += [
                    f"{own[own['task'] == t]['Q'].mean():.2f}" for t in TASK_NAMES
                ]
            lines.append(" & ".join(cells) + " \\\\")
        cells = ["", f"$\\rho$ ({count.replace('num_', '')})", ""]
        for model, (_, prefix) in labels.items():
            own = merged[merged["model"] == model]
            repo = own.groupby("repo_id").agg(
                Q=("Q", "mean"), n=(count, "first"), b=(col, "first")
            )
            rho, p_rho, cell = rho_cell(repo["n"], repo["Q"])
            cells.append(cell)
            rhos[""].append(rho)
            groups = [repo[repo["b"] == b]["Q"] for b in bins]
            macros += [
                cmd(f"{prefix}{title}Rho", signed(rho)),
                cmd(f"{prefix}{title}RhoP", fmt_p(p_rho)),
                cmd(f"{prefix}{title}KruskalP", fmt_p(stats.kruskal(*groups).pvalue)),
            ]
            for t in TASK_NAMES:
                task = own[own["task"] == t]
                rho_t, p_t, cell = rho_cell(task[count], task["Q"])
                cells.append(cell)
                rhos[TASK_NAMES[t]].append(rho_t)
                macros += [
                    cmd(f"{prefix}{title}{TASK_NAMES[t]}Rho", signed(rho_t)),
                    cmd(f"{prefix}{title}{TASK_NAMES[t]}RhoP", fmt_p(p_t)),
                ]
        lines.append(" & ".join(cells) + " \\\\")
        for key, values in rhos.items():
            macros += [
                cmd(f"{title}{key}RhoMin", signed(min(values))),
                cmd(f"{title}{key}RhoMax", signed(max(values))),
                cmd(f"{title}{key}RhoAbsMax", f"{max(map(abs, values)):.2f}"),
                cmd(f"{title}{key}RhoMean", signed(float(np.mean(values)))),
            ]
        if col != list(STRATA)[-1]:
            lines.append("\\midrule")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--rubric_version", default=RUBRIC_VERSION)
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="model=Label[=MacroPrefix], in table order",
    )
    parser.add_argument("--tex_dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Sampled dataset, for the strata table",
    )
    args = parser.parse_args()

    labels = {}
    for spec in args.labels:
        model, label, *prefix = spec.split("=")
        labels[model] = (label, prefix[0] if prefix else label)
    dims = COMMON_DIMS_BY_VERSION[args.rubric_version]
    unknown = set(dims) - set(DIM_NAMES)
    if unknown:
        raise SystemExit(f"No paper names for dimensions: {unknown}")
    assert set(DOMAIN_FIELD) == set(TASK_NAMES)

    rng = np.random.default_rng(0)
    macros = [
        "% Generated by reverse_se.evaluation.judge.paper_tables -- do not edit by hand."
    ]
    averaged = with_q(load_runs(args.scores, args.rubric_version), dims)
    averaged = averaged[averaged["model"].isin(labels)]
    lines_main = main_table(averaged, dims, labels, macros, rng)
    diagnostics(averaged, dims, labels, macros)

    args.tex_dir.mkdir(parents=True, exist_ok=True)
    (args.tex_dir / "judge_main.tex").write_text("\n".join(lines_main))
    written = ["judge_main.tex"]
    if len(args.scores) > 1:
        runs = [with_q(load_rows(p, args.rubric_version), dims) for p in args.scores]
        runs = [r[r["model"].isin(labels)] for r in runs]
        lines_rel = reliability_table(runs, dims, macros)
        ratings = pd.concat([r[dims] for r in runs]).stack()
        macros += [
            cmd("JudgeRatingMean", f"{ratings.mean():.2f}"),
            cmd("JudgeFivePct", f"{100 * (ratings == 5).mean():.0f}"),
        ]
        (args.tex_dir / "judge_reliability.tex").write_text("\n".join(lines_rel))
        written.append("judge_reliability.tex")
    if args.dataset:
        lines_strata = strata_table(averaged, load_strata(args.dataset), labels, macros)
        (args.tex_dir / "judge_strata.tex").write_text("\n".join(lines_strata))
        written.append("judge_strata.tex")
    (args.tex_dir / "judge_numbers.tex").write_text("\n".join(macros) + "\n")
    written.append("judge_numbers.tex")
    print(f"Wrote {', '.join(written)} to {args.tex_dir}")


if __name__ == "__main__":
    main()
