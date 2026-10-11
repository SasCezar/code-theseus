#!/usr/bin/env python3
"""LaTeX tables and numbers for the paper's agent code-generation results.

One round per generator: --round LABEL OUT STRUCTURE_CSV joins the runs
(OUT/results.jsonl, valid runs only, as in analyze_commit0.split_valid) with
the per-run structure metrics of that round (structure_commit0.py --csv,
with the generator's --docs-dir). The rounds are pooled; writes to --tex-dir:

  agent_main.tex       main results table (correctness, architecture, cost,
                       manipulation check)
  agent_extra.tex      all other metrics, for the appendix
  agent_libraries.tex  per-library held-out pass rate and module F1
  agent_numbers.tex    \\newcommand macros for the numbers quoted in the text
  agent_contract.tex   post-hoc moderation by contract coverage; per-library
                       data for scripts/R/agent_contract.R goes to
                       --contract-csv
  agent_generators.tex effect of Full per generator, condition x generator
                       interaction

Statistics, per metric, over libraries with both conditions:
  per library: mean over replicates and generators per condition; Delta = req_arch -
  intent_only; "Full >" / "Full <" count libraries where req_arch is higher
  / lower; Wilcoxon signed-rank test on the non-zero Deltas; linear mixed
  model on the runs (metric ~ condition, random intercept per library,
  REML), reporting the condition effect and its p-value; bootstrap 95% CI
  of the mean Delta (10,000 resamples of libraries, seed 0). Holm correction
  over the primary outcomes (held-out pass rate, module F1, symbol ARI).
  Per generator: the same paired test on that generator's runs, and the
  p-value of condition x generator in metric ~ condition * generator
  (random intercept per library; secondary, not adjusted).

Usage (on the machine with the runs; statsmodels isn't a project
dependency):
    uv run scripts/python/structure_commit0.py --out OUT --docs-dir DOCS \\
        --csv structure_runs.csv   # per round
    uv run --with statsmodels scripts/python/paper_commit0.py \\
        --round Gemma data/out/benchmark/commit0-gemma gemma.csv \\
        --round Q35=Qwen data/out/benchmark/commit0 qwen.csv --tex-dir paper/tables
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_commit0 import split_valid, steps_to_threshold
from scipy import stats

CONDITIONS = ("intent_only", "req_arch")
PRIMARY = ("held_out_pass_rate", "module_f1", "symbol_ari")
TIE = 1e-9


@dataclass
class Metric:
    key: str
    label: str
    group: str
    better: int  # +1 higher is better, -1 lower is better, 0 neither
    scale: float = 1.0
    digits: int = 2
    macro: str = ""  # prefix for agent_numbers.tex (default: from key)

    @property
    def prefix(self) -> str:
        words = {"50": "Fifty", "80": "Eighty"}
        name = self.macro or "".join(
            words.get(w, w.capitalize()) for w in self.key.split("_")
        )
        if not name.isalpha():  # LaTeX command names are letters only
            raise ValueError(f"{self.key}: set a macro name without digits")
        return name


MAIN = [
    Metric(
        "held_out_pass_rate",
        "Held-out tests passed (\\%)",
        "Correctness",
        1,
        100,
        1,
        "HeldOut",
    ),
    Metric(
        "module_f1", "Module F1, non-contract modules", "Architecture", 1, 1, 2, "ModF"
    ),
    Metric("module_precision", "\\quad precision", "Architecture", 1),
    Metric("module_recall", "\\quad recall", "Architecture", 1),
    Metric(
        "all_module_f1", "Module F1, all modules", "Architecture", 1, 1, 2, "AllModF"
    ),
    Metric("symbol_ari", "Symbol grouping (ARI)", "Architecture", 1, 1, 2, "Ari"),
    Metric("task_tokens", "Task prompt (k tokens)", "Cost", 0, 1e-3, 1),
    Metric("agent_steps", "Agent steps", "Cost", -1, 1, 0),
    Metric("output_tokens", "Output tokens (k)", "Cost", -1, 1e-3, 0, "OutTok"),
    Metric("input_tokens", "Input tokens (M)", "Cost", -1, 1e-6, 1, "InTok"),
    Metric(
        "doc_module_recall",
        "Artefact modules built",
        "Use of the artefacts",
        1,
        1,
        2,
        "DocRecall",
    ),
]
EXTRA = [
    Metric("visible_pass_rate", "Visible tests passed (\\%)", "Correctness", 1, 100, 1),
    Metric("all_pass_rate", "All tests passed (\\%)", "Correctness", 1, 100, 1),
    Metric("placement_accuracy", "Symbol placement accuracy", "Architecture", 1),
    Metric("edge_precision", "Import-edge precision", "Architecture", 1),
    Metric("edge_recall", "Import-edge recall", "Architecture", 1),
    Metric("edge_f1", "Import-edge F1", "Architecture", 1, macro="EdgeF"),
    Metric("agent_modules", "Modules created", "Architecture", 0, 1, 1),
    Metric("agent_loc", "Lines of code (k)", "Architecture", 0, 1e-3, 2),
    Metric("wall_time_seconds", "Wall time (h)", "Cost and behaviour", -1, 1 / 3600, 2),
    Metric(
        "efficiency",
        "Held-out pp per 100k output tokens",
        "Cost and behaviour",
        1,
        1,
        2,
    ),
    Metric(
        "finished_by_tool",
        "Finished by the agent (\\%)",
        "Cost and behaviour",
        1,
        100,
        1,
    ),
    Metric(
        "steps_to_50", "Steps to 50\\% visible passing", "Cost and behaviour", -1, 1, 0
    ),
    Metric(
        "steps_to_80", "Steps to 80\\% visible passing", "Cost and behaviour", -1, 1, 0
    ),
    Metric(
        "search_attempts", "Searches for the reference", "Cost and behaviour", -1, 1, 1
    ),
]


# ── data ──────────────────────────────────────────────────────────────────────


def load_round(args, generator: str, out: Path, structure_csv: Path):
    df = pd.DataFrame(
        [
            json.loads(line)
            for line in (out / "results.jsonl").read_text().splitlines()
            if line.strip()
        ]
    )
    df = df[(df["scaffold"] == args.scaffold) & ~df["name"].isin(args.exclude)]
    if args.model:
        df = df[df["model"] == args.model]
    valid, bad = split_valid(df)
    print(f"{generator}: {len(df)} runs, {len(valid)} valid, {len(bad)} excluded")
    if len(bad):
        print(bad[["name", "condition", "replicate", "reason"]].to_string(index=False))
    valid = valid.copy()
    valid["generator"] = generator
    valid["replicate"] = valid["replicate"].astype(int)
    valid["finished_by_tool"] = valid["finished_by_tool"].astype(float)
    valid["efficiency"] = (
        100 * valid["held_out_pass_rate"] / (valid["output_tokens"] / 1e5)
    )
    for share in (0.5, 0.8):
        valid[f"steps_to_{int(share * 100)}"] = [
            steps_to_threshold(
                out
                / "runs"
                / r.model.replace("/", "__")
                / r.scaffold
                / r.name
                / r.condition
                / f"{r.replicate:02d}",
                int(r.visible_total or 0),
                share,
            )
            for r in valid.itertuples()
        ]
    structure = pd.read_csv(structure_csv).drop(columns=["held_out_pass_rate"])
    runs = valid.merge(structure, on=["name", "condition", "replicate"], how="left")
    missing = runs["symbol_ari"].isna() & runs["agent_modules"].isna()
    if missing.any():
        raise SystemExit(
            f"{generator}: {missing.sum()} runs have no structure metrics:"
            " rerun structure_commit0.py"
        )
    return runs, len(df)


def load_runs(args) -> tuple[pd.DataFrame, int]:
    rounds = [
        load_round(args, label.split("=")[0], Path(out), Path(csv))
        for label, out, csv in args.round
    ]
    return (
        pd.concat([r for r, _ in rounds], ignore_index=True),
        sum(n for _, n in rounds),
    )


# ── statistics ────────────────────────────────────────────────────────────────


def mixed_model(data: pd.DataFrame, key: str) -> tuple[float, float, float, float]:
    """Condition effect (req_arch - intent_only), its 95% CI and p-value."""
    import statsmodels.formula.api as smf

    if data[key].nunique() < 2:
        return (np.nan,) * 4
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"{key} ~ C(condition)", data, groups=data["name"]).fit(
                reml=True
            )
    except (ValueError, np.linalg.LinAlgError):
        return (np.nan,) * 4
    term = next(t for t in fit.params.index if "req_arch" in t)
    low, high = fit.conf_int().loc[term]
    return fit.params[term], low, high, fit.pvalues[term]


def rank_biserial(diffs: pd.Series) -> float:
    """Matched-pairs rank-biserial correlation of non-zero paired differences:
    (sum of ranks of positive - sum of ranks of negative) / sum of all ranks,
    ranks of |difference|. In [-1, 1]; > 0 when Full tends to be higher."""
    if not len(diffs):
        return np.nan
    ranks = stats.rankdata(diffs.abs())
    return (
        ranks[diffs.to_numpy() > 0].sum() - ranks[diffs.to_numpy() < 0].sum()
    ) / ranks.sum()


def shown_diff(intent: float, full: float, m: Metric) -> float:
    return round(float(fmt(full, m.digits)) - float(fmt(intent, m.digits)), m.digits)


def compare(runs: pd.DataFrame, m: Metric) -> dict:
    data = runs[["name", "condition", m.key]].dropna().copy()
    data[m.key] = data[m.key].astype(float) * m.scale
    lib = data.groupby(["name", "condition"])[m.key].mean().unstack()
    if not set(CONDITIONS) <= set(lib):
        return {"n": 0}
    lib = lib.dropna(subset=list(CONDITIONS))
    data = data[data["name"].isin(lib.index)]
    diff = lib["req_arch"] - lib["intent_only"]
    nonzero = diff[diff.abs() > TIE]
    boots = (
        np.random.default_rng(0).choice(diff.to_numpy(), (10_000, len(diff))).mean(1)
    )
    effect, low, high, p_mixed = mixed_model(data, m.key)
    return {
        "n": len(lib),
        "intent": lib["intent_only"].mean(),
        "full": lib["req_arch"].mean(),
        "diff": diff.mean(),
        # What the tables print: the difference of the rounded means, so a
        # row reads consistently (Intent, Full, Delta).
        "diff_shown": shown_diff(lib["intent_only"].mean(), lib["req_arch"].mean(), m),
        "ci_low": np.percentile(boots, 2.5),
        "ci_high": np.percentile(boots, 97.5),
        "higher": int((diff > TIE).sum()),
        "lower": int((diff < -TIE).sum()),
        "p_wilcoxon": stats.wilcoxon(nonzero).pvalue if len(nonzero) else np.nan,
        "r_rb": rank_biserial(nonzero),
        "effect": effect,
        "effect_low": low,
        "effect_high": high,
        "p_mixed": p_mixed,
        "per_library": lib,
    }


def library_moderators(runs: pd.DataFrame) -> pd.DataFrame:
    """Per library: contract coverage (share of reference modules the
    contract names) and documentation coverage (share of non-contract
    reference modules the requirements and architecture artefacts name, mean
    over the generators; undefined when the contract names every module)."""
    lib = runs.groupby(["name", "generator"])[
        ["contract_modules", "reference_modules", "documented_modules"]
    ].first()
    lib["cov"] = lib["contract_modules"] / lib["reference_modules"]
    rest = lib["reference_modules"] - lib["contract_modules"]
    lib["doc_cov"] = (lib["documented_modules"] / rest).where(rest > 0)
    return lib.groupby("name")[
        [
            "contract_modules",
            "reference_modules",
            "documented_modules",
            "cov",
            "doc_cov",
        ]
    ].mean()


def interaction_p(data: pd.DataFrame, key: str) -> float:
    """p-value of condition x coverage in metric ~ condition * cov, random
    intercept per library."""
    import statsmodels.formula.api as smf

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(
                f"{key} ~ C(condition) * cov", data, groups=data["name"]
            ).fit(reml=True)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan
    return fit.pvalues[next(t for t in fit.params.index if ":cov" in t)]


def moderation(
    results: dict[str, dict], runs: pd.DataFrame, metrics: list[Metric]
) -> tuple[dict[str, dict], float, pd.DataFrame]:
    """Does the Full - Intent difference depend on how much of the structure
    the contract already gives away? Post-hoc, so not in the Holm family.
    Per metric: Spearman correlation of the per-library difference with
    contract and documentation coverage, the mean difference below / at or
    above the median coverage (all libraries), and the condition x coverage
    interaction in the mixed model."""
    mods = library_moderators(runs)
    median = mods["cov"].median()
    out, long = {}, []
    for m in metrics:
        lib = results[m.key]["per_library"].join(mods, how="inner")
        lib["diff"] = lib["req_arch"] - lib["intent_only"]
        rho, p = stats.spearmanr(lib["diff"], lib["cov"])
        docs = lib.dropna(subset=["doc_cov"])
        doc_rho, doc_p = stats.spearmanr(docs["diff"], docs["doc_cov"])
        halves = {}
        for half, part in (
            ("Low", lib[lib["cov"] < median]),
            ("High", lib[lib["cov"] >= median]),
        ):
            nonzero = part["diff"][part["diff"].abs() > TIE]
            halves[half] = {
                "n": len(part),
                "diff": part["diff"].mean(),
                "p": stats.wilcoxon(nonzero).pvalue if len(nonzero) else np.nan,
            }
        data = runs[["name", "condition", m.key]].dropna().copy()
        data[m.key] = data[m.key].astype(float) * m.scale
        data = data.join(mods["cov"], on="name")
        out[m.key] = {
            "n": len(lib),
            "rho": rho,
            "p": p,
            "doc_n": len(docs),
            "doc_rho": doc_rho,
            "doc_p": doc_p,
            "halves": halves,
            "p_interaction": interaction_p(data, m.key),
        }
        long.append(
            lib.reset_index().assign(
                outcome=m.key, rho=rho, rho_p=p, doc_rho=doc_rho, doc_rho_p=doc_p
            )
        )
    frame = pd.concat(long, ignore_index=True).rename(
        columns={"intent_only": "intent", "req_arch": "full"}
    )
    return out, median, frame


def generator_p(data: pd.DataFrame, key: str) -> tuple[float, float]:
    """p-values of the generator (metric ~ condition + generator) and of
    condition x generator (metric ~ condition * generator), random
    intercept per library."""
    import statsmodels.formula.api as smf

    out = []
    for formula, mark in (("+", "C(generator)"), ("*", ":C(generator)")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    f"{key} ~ C(condition) {formula} C(generator)",
                    data,
                    groups=data["name"],
                ).fit(reml=True)
        except (ValueError, np.linalg.LinAlgError):
            out.append(np.nan)
            continue
        terms = [
            t
            for t in fit.params.index
            if mark in t and (mark.startswith(":") or ":" not in t)
        ]
        out.append(fit.pvalues[terms[0]] if len(terms) == 1 else np.nan)
    return out[0], out[1]


def by_generator(
    runs: pd.DataFrame, metrics: list[Metric]
) -> tuple[list[str], dict[str, dict]]:
    """Per metric: the paired comparison within each generator, and the
    generator and condition x generator p-values on all runs."""
    generators = list(dict.fromkeys(runs["generator"]))
    out = {}
    for m in metrics:
        data = runs[["name", "condition", "generator", m.key]].dropna().copy()
        data[m.key] = data[m.key].astype(float) * m.scale
        p_gen, p_inter = generator_p(data, m.key)
        out[m.key] = {
            "per": {g: compare(runs[runs["generator"] == g], m) for g in generators},
            "p_generator": p_gen,
            "p_interaction": p_inter,
        }
    return generators, out


GENERATOR_LABELS = {
    "held_out_pass_rate": "Held-out tests (pp)",
    "module_f1": "Module F1",
    "symbol_ari": "Symbol grouping",
}


def generator_table(
    generators: list[str],
    gen: dict[str, dict],
    metrics: list[Metric],
    runs: pd.DataFrame,
    tex_names: dict[str, str],
) -> str:
    cols = "r" * len(generators)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Per generator: $\\Delta$ (\\cFull $-$ \\cIntent) over library"
        " means, Wilcoxon $p$ in parentheses. Gen.: "
        "between generators; Int.:"
        " condition $\\times$ generator (mixed models, unadjusted).}",
        "\\label{tab:agent_generators}",
        f"\\begin{{tabular}}{{l{cols}rr}}",
        "\\toprule",
        "\\textbf{Metric}"
        + "".join(f" & \\textbf{{{tex_names[g]}}}" for g in generators)
        + " & \\textbf{Gen.} & \\textbf{Int.} \\\\",
        "\\midrule",
    ]
    for m in metrics:
        g = gen[m.key]
        cells = [GENERATOR_LABELS.get(m.key, m.label)]
        for name in generators:
            r = g["per"][name]
            p = fmt_p(r["p_wilcoxon"])
            cells.append(f"{signed(r['diff_shown'], m.digits)} ({p})")
        cells += [fmt_p(g["p_generator"]), fmt_p(g["p_interaction"])]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    cells = ["Artefact cov. (\\%)"]
    for name in generators:
        lib = runs[runs["generator"] == name].groupby("name")[
            ["contract_modules", "reference_modules", "documented_modules"]
        ]
        lib = lib.first()
        rest = lib["reference_modules"] - lib["contract_modules"]
        cells.append(
            fmt(100 * (lib["documented_modules"] / rest).where(rest > 0).mean(), 0)
        )
    lines.append(" & ".join(cells) + " & -- & -- \\\\")
    full = runs[runs["condition"] == "req_arch"]
    rows = {
        "Task prompt, \\cFull (k tok.)": (full, "task_tokens", 1e-3, 1),
        "Artefact modules built, \\cFull": (full, "doc_module_recall", 1, 2),
    }
    for label, (data, key, scale, digits) in rows.items():
        cells = [label]
        for name in generators:
            own = data[data["generator"] == name]
            value = own.groupby("name")[key].mean().mean() * scale
            cells.append(fmt(value, digits))
        lines.append(" & ".join(cells) + " & -- & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def generator_macros(
    generators: list[str],
    gen: dict[str, dict],
    metrics: list[Metric],
    macro_names: dict[str, str],
) -> str:
    def cmd(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"

    out = []
    for m in metrics:
        g = gen[m.key]
        for name in generators:
            r = g["per"][name]
            name = macro_names[name]
            out += [
                cmd(f"{m.prefix}Diff{name}", signed(r["diff_shown"], m.digits)),
                cmd(f"{m.prefix}Wilcoxon{name}", p_text(r["p_wilcoxon"])),
                cmd(f"{m.prefix}Higher{name}", str(r["higher"])),
                cmd(f"{m.prefix}Libs{name}", str(r["n"])),
            ]
        out += [
            cmd(f"{m.prefix}GenP", p_text(g["p_generator"])),
            cmd(f"{m.prefix}GenInterP", p_text(g["p_interaction"])),
        ]
    return "\n".join(out) + "\n"


def library_frame(runs: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    """Per library, outcome and generator ("Pooled" for both): mean over
    replicates (and generators) per condition, and their difference."""
    frames = []
    groups = [("Pooled", runs)] + list(runs.groupby("generator", sort=False))
    for generator, data in groups:
        for m in metrics:
            lib = compare(data, m)["per_library"]
            frames.append(
                lib.reset_index().assign(
                    outcome=m.key,
                    generator=generator,
                    diff=lib["req_arch"].to_numpy() - lib["intent_only"].to_numpy(),
                )
            )
    return pd.concat(frames, ignore_index=True).rename(
        columns={"intent_only": "intent", "req_arch": "full"}
    )


COVERAGE_OUTCOMES = ("module_f1", "module_recall", "held_out_pass_rate")


def coverage_units(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per library and generator: the gain of Full over Intent
    (mean over replicates) per outcome, the documentation coverage of that
    generator's artefacts (share of non-contract reference modules they
    name) and their length (task text in Full, k tokens)."""
    units = runs.groupby(["name", "generator"])[
        ["contract_modules", "reference_modules", "documented_modules"]
    ].first()
    rest = units["reference_modules"] - units["contract_modules"]
    units["doc_cov"] = (units["documented_modules"] / rest).where(rest > 0)
    full = runs[runs["condition"] == "req_arch"]
    units["length"] = full.groupby(["name", "generator"])["task_tokens"].mean() / 1e3
    for key in COVERAGE_OUTCOMES:
        scale = 100 if key == "held_out_pass_rate" else 1
        means = runs.groupby(["name", "generator", "condition"])[key].mean().unstack()
        units[f"gain_{key}"] = scale * (means["req_arch"] - means["intent_only"])
    return units.reset_index()


def coverage_models(units: pd.DataFrame) -> dict[str, dict]:
    """gain ~ z(predictor) + (1 | library), for documentation coverage,
    length, and both together; the random intercept per library compares
    the generators within a library. Exploratory, not adjusted."""
    import statsmodels.formula.api as smf

    out = {}
    for key in COVERAGE_OUTCOMES:
        data = units.dropna(subset=[f"gain_{key}", "doc_cov", "length"]).copy()
        for col in ("doc_cov", "length"):
            data[f"z_{col}"] = (data[col] - data[col].mean()) / data[col].std()
        fits = {}
        for name, formula in (
            ("cov", "z_doc_cov"),
            ("len", "z_length"),
            ("both", "z_doc_cov + z_length"),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    f"gain_{key} ~ {formula}", data, groups=data["name"]
                ).fit(reml=True)
            fits[name] = {t: (fit.params[t], fit.pvalues[t]) for t in fit.params.index}
        out[key] = {
            "n": len(data),
            "libs": data["name"].nunique(),
            "sd": {col: data[col].std() for col in ("doc_cov", "length")},
            **fits,
        }
    return out


def coverage_macros(models: dict[str, dict]) -> str:
    def cmd(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"

    prefixes = {
        "module_f1": ("ModF", 3),
        "module_recall": ("ModuleRecall", 3),
        "held_out_pass_rate": ("HeldOut", 1),
    }
    first = next(iter(models.values()))
    out = [
        cmd("CovGainUnits", str(first["n"])),
        cmd("CovGainLibs", str(first["libs"])),
        # SDs behind the per-SD betas: coverage in pp, length in k tokens.
        cmd("DocCovSD", f"{100 * first['sd']['doc_cov']:.0f}"),
        cmd("LengthSD", f"{first['sd']['length']:.1f}"),
    ]
    for key, r in models.items():
        prefix, digits = prefixes[key]
        for name, term, model in (
            ("CovGain", "z_doc_cov", "cov"),
            ("LenGain", "z_length", "len"),
            ("CovGainAdj", "z_doc_cov", "both"),
        ):
            beta, p = r[model][term]
            out += [
                cmd(f"{prefix}{name}Beta", signed(beta, digits)),
                cmd(f"{prefix}{name}P", p_text(p)),
            ]
    return "\n".join(out) + "\n"


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


# ── LaTeX ─────────────────────────────────────────────────────────────────────


def fmt(value: float, digits: int) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def signed(value: float, digits: int) -> str:
    """Signed number with a proper minus sign, usable in text and math."""
    if pd.isna(value):
        return "--"
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0:
        return text
    return ("\\ensuremath{+}" if value > 0 else "\\ensuremath{-}") + text


def p_text(p: float) -> str:
    if pd.isna(p):
        return "--"
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "--"
    text = "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")
    return f"\\textbf{{{text}}}" if p < 0.05 else text


def fmt_delta(diff: float, m: Metric, significant: bool) -> str:
    """Change with the paper's arrows: direction of the change, green if it
    is an improvement, red if not. Metrics with no better direction get a
    plain signed number instead."""
    text = f"{abs(diff):.{m.digits}f}"
    if pd.isna(diff) or float(text) == 0:
        return fmt(0.0 if not pd.isna(diff) else diff, m.digits)
    if m.better == 0:
        body = signed(diff, m.digits)
        return f"\\textbf{{{body}}}" if significant else body
    arrow = "\\up" if diff > 0 else "\\down"
    tone = "good" if diff * m.better > 0 else "bad"
    body = f"\\textbf{{{text}}}" if significant else text
    return f"{arrow}{tone}{{{body}}}"


def arrow_label(m: Metric) -> str:
    return m.label + {1: " $\\uparrow$", -1: " $\\downarrow$", 0: ""}[m.better]


def library_counts(r: dict, m: Metric) -> str:
    """Libraries where Full is better / tied / worse, as three cells joined
    by the column separator so the slashes align: better in green, worse in
    red. Metrics with no better direction: higher / tied / lower, plain."""
    ties = r["n"] - r["higher"] - r["lower"]
    if m.better == 0:
        return f"{r['higher']} & {ties} & {r['lower']}"
    better, worse = (
        (r["higher"], r["lower"]) if m.better > 0 else (r["lower"], r["higher"])
    )
    return (
        f"\\textcolor{{better}}{{{better}}} & {ties} & \\textcolor{{worse}}{{{worse}}}"
    )


def n_cell(m: Metric, r: dict) -> str:
    mark = N_MARKS.get(m.key)
    # \rlap: the mark takes no width, so n stays aligned with the unmarked rows
    return f"{r['n']}\\rlap{{$^{{{mark}}}$}}" if mark else str(r["n"])


def ci_cell(r: dict, m: Metric) -> str:
    return f"[{signed(r['ci_low'], m.digits)}, {signed(r['ci_high'], m.digits)}]"


# Metrics defined on fewer libraries: a mark on n, explained under the table
N_MARKS = {
    "module_f1": "*",
    "module_precision": "*",
    "module_recall": "*",
    "symbol_ari": "\\dagger",
    "doc_module_recall": "\\ddagger",
}
N_NOTES = {
    "*": "$^{*}$Without the libraries whose contract names every module.",
    "\\dagger": "$^{\\dagger}$Without the libraries whose two implementations"
    " share fewer than two classes or functions outside the contract.",
    "\\ddagger": "$^{\\ddagger}$Without the libraries whose requirements and"
    " architecture name no module outside the contract.",
}


def summary_table(
    results: list[tuple[Metric, dict]],
    caption: str,
    label: str,
    holm_p: dict[str, float] | None = None,
    effect: bool = True,
) -> str:
    """Bootstrap CI of Delta, and the p-values of the Wilcoxon test and the
    mixed model; with holm_p, also the Holm-adjusted p-value (primary
    outcomes only); with effect, the rank-biserial r."""
    tests = ["Wilcoxon"] + (["Holm"] if holm_p is not None else []) + ["Mixed"]
    marks = dict.fromkeys(
        N_MARKS[m.key] for m, r in results if r["n"] and m.key in N_MARKS
    )
    # B / T / W spans three columns whose separator is the slash
    width = 9 + effect + len(tests)
    last = 9 + effect
    lines = [
        "\\begin{table*}[htbp!]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4.5pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{lrrrrrr@{{\\ /\\ }}c@{{\\ /\\ }}l{'r' * (effect + len(tests))}}}",
        "\\toprule",
        " & & \\multicolumn{4}{c}{\\textbf{Mean over libraries}}"
        " & \\multicolumn{3}{c}{\\textbf{Libraries}}"
        + (" & \\textbf{Effect}" if effect else "")
        + f" & \\multicolumn{{{len(tests)}}}{{c}}{{\\textbf{{$p$-value}}}} \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-9}"
        + (" \\cmidrule(lr){10-10}" if effect else "")
        + f" \\cmidrule(lr){{{last + 1}-{width}}}",
        "\\textbf{Metric} & $n$ & \\textbf{\\cIntent} & \\textbf{\\cFull} & $\\Delta$"
        " & \\textbf{95\\% CI} & \\multicolumn{3}{c}{\\textbf{B / T / W}}"
        + (" & $r$" if effect else "")
        + "".join(f" & \\textbf{{{t}}}" for t in tests)
        + " \\\\",
    ]
    group = None
    for m, r in results:
        if m.group != group:
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{width}}}{{l}}{{\\textit{{{m.group}}}}} \\\\")
            group = m.group
        if not r["n"]:
            continue
        significant = r["p_wilcoxon"] < 0.05
        p_cells = [fmt_p(r["p_wilcoxon"])]
        if holm_p is not None:
            p_cells.append(fmt_p(holm_p[m.key]) if m.key in holm_p else "--")
        p_cells.append(fmt_p(r["p_mixed"]))
        lines.append(
            f"{arrow_label(m)} & {n_cell(m, r)} & {fmt(r['intent'], m.digits)}"
            f" & {fmt(r['full'], m.digits)} & {fmt_delta(r['diff_shown'], m, significant)}"
            f" & {ci_cell(r, m)} & {library_counts(r, m)}"
            + (f" & {signed(r['r_rb'], 2)}" if effect else "")
            + f" & {' & '.join(p_cells)} \\\\"
        )
    lines.append("\\bottomrule")
    lines += [
        f"\\multicolumn{{{width}}}{{@{{}}l}}{{\\footnotesize {N_NOTES[k]}}} \\\\"
        for k in marks
    ]
    lines += [
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines)


def library_table(
    results: dict[str, dict],
    runs: pd.DataFrame,
    lite: set[str],
    scope: str,
    average: str,
    label: str = "tab:agent_libraries",
) -> str:
    shown = [
        m for m in MAIN if m.key in ("held_out_pass_rate", "module_f1", "all_module_f1")
    ]
    tests = runs.groupby("name")["held_out_total"].first()
    names = sorted(tests.index)
    header_groups = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\textbf{{{title}}}}}"
        for title in (
            "Held-out passed (\\%)",
            "Module F1, non-contract",
            "Module F1, all",
        )
    )
    lines = [
        "\\begin{table*}[htbp!]",
        "\\centering",
        "\\small",
        f"\\caption{{Per-library results, \\textbf{{{scope}}}: mean over"
        f" {average}. \\cFull: intent, requirements and C4 architecture; \\cIntent:"
        " intent only. $^{\\dagger}$\\textsc{Commit0-Lite}. --: undefined,"
        " because the contract names every module of the library.}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lr" + "rrr" * len(shown) + "}",
        "\\toprule",
        f" & & {header_groups} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}",
        "\\textbf{Library} & \\textbf{Held-out tests}"
        + " & \\textbf{\\cIntent} & \\textbf{\\cFull} & $\\Delta$" * len(shown)
        + " \\\\",
        "\\midrule",
    ]
    for name in names:
        cells = [
            f"{name}{'$^{\\dagger}$' if name in lite else ''}",
            str(int(tests[name])),
        ]
        for m in shown:
            lib = results[m.key]["per_library"]
            if name in lib.index:
                i, f = lib.loc[name, "intent_only"], lib.loc[name, "req_arch"]
                cells += [
                    fmt(i, m.digits),
                    fmt(f, m.digits),
                    fmt_delta(shown_diff(i, f, m), m, False),
                ]
            else:
                cells += ["--", "--", "--"]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    cells = ["Mean", ""]
    for m in shown:
        r = results[m.key]
        cells += [
            fmt(r["intent"], m.digits),
            fmt(r["full"], m.digits),
            fmt_delta(r["diff_shown"], m, r["p_wilcoxon"] < 0.05),
        ]
    lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines)


def rho_cell(rho: float, p: float) -> str:
    return f"{signed(rho, 2)} ({fmt_p(p)})"


# Outcomes of the moderation table (columns), with short headers.
MODERATION_COLUMNS = {
    "module_f1": "Module F1",
    "held_out_pass_rate": "Held-out (pp)",
}


def moderation_table(
    mod: dict[str, dict],
    cov_models: dict[str, dict],
    metrics: list[Metric],
) -> str:
    """One row per statistic, grouped by moderator, one column per outcome:
    does the gain of Full depend on contract coverage, on documentation
    coverage or on artefact length? Compact: the moderator labels its first
    row instead of taking a row of its own, the two median halves share a
    row, and the library counts go in the column headers (the contract
    coverage range is quoted in the text)."""
    shown = [m for m in metrics if m.key in MODERATION_COLUMNS]
    beta_digits = {"module_f1": 3, "held_out_pass_rate": 1}

    def row(moderator: str, label: str, cell) -> str:
        return " & ".join([moderator, label] + [cell(m) for m in shown]) + " \\\\"

    def beta(model: str, term: str):
        def cell(m: Metric) -> str:
            b, p = cov_models[m.key][model][term]
            return f"{signed(b, beta_digits[m.key])} ({fmt_p(p)})"

        return cell

    def halves(m: Metric) -> str:
        h = mod[m.key]["halves"]
        return f"{signed(h['Low']['diff'], m.digits)} / {signed(h['High']['diff'], m.digits)}"

    header = "".join(
        f" & \\textbf{{{MODERATION_COLUMNS[m.key]}}} ({mod[m.key]['n']})" for m in shown
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        # Pooled over generators, as all results unless stated otherwise.
        "\\caption{Gain of \\cFull ($p$ in parentheses)."
        " Contract cov.: share of modules the contract"
        " names; artefact cov.: share of non-contract modules the artefacts name."
        " $\\beta$: within library.}",
        "\\label{tab:agent_contract}",
        f"\\begin{{tabular}}{{ll{'r' * len(shown)}}}",
        "\\toprule",
        " &" + header + " \\\\",
        "\\midrule",
        row(
            "Contract cov.",
            "Spearman $\\rho$",
            lambda m: rho_cell(mod[m.key]["rho"], mod[m.key]["p"]),
        ),
        row("", "Gain $<$ / $>$ median", halves),
        row("", "Interaction $p$", lambda m: fmt_p(mod[m.key]["p_interaction"])),
        "\\midrule",
        row(
            "Artefact cov.",
            "Spearman $\\rho$",
            lambda m: rho_cell(mod[m.key]["doc_rho"], mod[m.key]["doc_p"]),
        ),
        row("", "$\\beta$ per SD", beta("cov", "z_doc_cov")),
        row("", "$\\beta$ per SD, length-adj.", beta("both", "z_doc_cov")),
        "\\midrule",
        row("Length", "$\\beta$ per SD", beta("len", "z_length")),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def moderation_macros(
    mod: dict[str, dict], metrics: list[Metric], median: float, runs: pd.DataFrame
) -> str:
    def cmd(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"

    mods = library_moderators(runs)
    out = [
        cmd("ContractCovMedian", f"{100 * median:.0f}"),
        cmd("ContractCovMin", f"{100 * mods['cov'].min():.0f}"),
        cmd("ContractCovMax", f"{100 * mods['cov'].max():.0f}"),
        cmd("ContractFullLibs", str(int((mods["cov"] >= 1 - TIE).sum()))),
    ]
    for m in metrics:
        r = mod[m.key]
        values = {
            "CovN": str(r["n"]),
            "CovRho": signed(r["rho"], 2),
            "CovP": p_text(r["p"]),
            "DocCovN": str(r["doc_n"]),
            "DocCovRho": signed(r["doc_rho"], 2),
            "DocCovP": p_text(r["doc_p"]),
            "InteractionP": p_text(r["p_interaction"]),
        }
        for half, h in r["halves"].items():
            values[f"{half}CovN"] = str(h["n"])
            values[f"{half}CovDiff"] = signed(h["diff"], m.digits)
            values[f"{half}CovP"] = p_text(h["p"])
        out += [cmd(m.prefix + suffix, v) for suffix, v in values.items()]
    return "\n".join(out) + "\n"


def number_macros(
    results: dict[str, dict], runs: pd.DataFrame, total: int, adjusted: dict
) -> str:
    def cmd(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"

    out = [
        "% Generated by scripts/python/paper_commit0.py -- do not edit by hand.",
        cmd("AgentLibs", str(runs["name"].nunique())),
        cmd("AgentRuns", str(len(runs))),
        cmd("AgentRunsTotal", str(total)),
        cmd("AgentRunsExcluded", str(total - len(runs))),
        cmd("AgentReplicates", str(runs["replicate"].max())),
        cmd("AgentGenerators", str(runs["generator"].nunique())),
    ]
    sd = (
        runs.groupby(["name", "condition", "generator"])["held_out_pass_rate"]
        .std()
        .dropna()
    )
    out.append(cmd("HeldOutReplicateSD", f"{100 * sd.mean():.1f}"))
    # Runs that used the whole step budget, overall and per condition.
    at_cap = runs["agent_steps"] >= runs["max_iterations"]
    out += [
        cmd("AgentMaxSteps", f"{int(runs['max_iterations'].max()):,}"),
        cmd("AtCapPct", f"{100 * at_cap.mean():.0f}"),
        cmd(
            "AtCapIntent",
            f"{100 * at_cap[runs['condition'] == 'intent_only'].mean():.0f}",
        ),
        cmd("AtCapFull", f"{100 * at_cap[runs['condition'] == 'req_arch'].mean():.0f}"),
    ]
    for m in MAIN + EXTRA:
        r = results[m.key]
        if not r["n"]:
            continue
        d = m.digits
        values = {
            "Libs": str(r["n"]),
            "Intent": fmt(r["intent"], d),
            "Full": fmt(r["full"], d),
            "Diff": signed(r["diff_shown"], d),
            "CILow": signed(r["ci_low"], d),
            "CIHigh": signed(r["ci_high"], d),
            "Higher": str(r["higher"]),
            "Lower": str(r["lower"]),
            "Ties": str(r["n"] - r["higher"] - r["lower"]),
            "Wilcoxon": p_text(r["p_wilcoxon"]),
            "R": signed(r["r_rb"], 2),
            "Effect": signed(r["effect"], d + 1),
            "EffectLow": signed(r["effect_low"], d + 1),
            "EffectHigh": signed(r["effect_high"], d + 1),
            "Mixed": p_text(r["p_mixed"]),
        }
        if m.key in adjusted:
            values["Holm"] = p_text(adjusted[m.key])
        if r["intent"]:
            values["Pct"] = signed(100 * (r["full"] / r["intent"] - 1), 0)
        out += [cmd(m.prefix + suffix, v) for suffix, v in values.items()]
    # Spread of the per-library held-out effect.
    lib = results["held_out_pass_rate"]["per_library"]
    diff = lib["req_arch"] - lib["intent_only"]
    out += [
        cmd("HeldOutMaxGain", signed(diff.max(), 1)),
        cmd("HeldOutMaxGainLib", diff.idxmax()),
        cmd("HeldOutMaxLoss", signed(diff.min(), 1)),
        cmd("HeldOutMaxLossLib", diff.idxmin()),
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--round",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "OUT", "STRUCTURE_CSV"),
        help="One generator round; LABEL is NAME or NAME=MACRO, with a"
        " letters-only macro suffix when NAME isn't one; tables print the"
        " generator as \\m<MACRO>, defined in the manuscript",
    )
    parser.add_argument("--scaffold", default="contract")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--contract-csv",
        type=Path,
        default=Path("data/out/benchmark/agent_contract_libraries.csv"),
    )
    parser.add_argument(
        "--libraries-csv",
        type=Path,
        default=Path("data/out/benchmark/agent_libraries_generators.csv"),
        help="Per library and generator, for scripts/R/agent_libraries.R",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("data/raw/commit0/commit0_instances.jsonl"),
    )
    parser.add_argument("--exclude", nargs="+", default=[], metavar="NAME")
    parser.add_argument("--tex-dir", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()

    runs, total = load_runs(args)
    lite = {
        i["name"]
        for i in map(json.loads, args.instances.read_text().splitlines())
        if i.get("lite")
    }
    results = {m.key: compare(runs, m) for m in MAIN + EXTRA}
    adjusted = holm({k: results[k]["p_wilcoxon"] for k in PRIMARY})

    args.tex_dir.mkdir(parents=True, exist_ok=True)

    def summary_caption(scope: str, which: str, average: str) -> str:
        """Caption shared by the appendix summary tables; only the scope
        (bold), the metric set and what is averaged differ."""
        return (
            f"Agent results, \\textbf{{{scope}}}: {which} metrics, exploratory"
            f" ($p$ not corrected). Mean over libraries of each library's mean"
            f" over {average}; $n$: libraries. \\cFull: intent, requirements and C4"
            " architecture; \\cIntent: intent only. $\\Delta$: \\cFull $-$ \\cIntent;"
            " $\\blacktriangle$/$\\blacktriangledown$: \\cFull higher/lower, a"
            " plain signed number when no direction is better; bold when"
            " Wilcoxon $p < .05$; 95\\% CI: bootstrap over libraries. B / T / W:"
            " libraries where \\cFull is better /"
            " tied / worse (higher / tied / lower when no direction is better)."
            " $r$: matched-pairs rank-biserial correlation ($-1$ to 1, positive"
            " when \\cFull is higher). Wilcoxon: signed-rank test over libraries."
            " Mixed: linear mixed model on the runs, random intercept per"
            " library."
        )

    (args.tex_dir / "agent_main.tex").write_text(
        summary_table(
            [(m, results[m.key]) for m in MAIN],
            "Agent results, averaged over both generators."
            " $\\Delta$: \\cFull $-$ \\cIntent, bold when Wilcoxon $p < .05$, with"
            " bootstrap CI. B / T / W: libraries where \\cFull is better / tied /"
            " worse. Holm: over the three primary outcomes; Mixed: mixed model"
            " on all runs.",
            "tab:agent_main",
            holm_p=adjusted,
            effect=False,
        )
    )
    macro_names = {
        label.split("=")[0]: label.split("=")[-1] for label, _, _ in args.round
    }
    # Tables name generators by the manuscript's label macros (\m<MACRO>)
    tex_names = {g: f"\\m{name}" for g, name in macro_names.items()}
    averaged = " and ".join(tex_names.values()) + " averaged"
    (args.tex_dir / "agent_extra.tex").write_text(
        summary_table(
            [(m, results[m.key]) for m in EXTRA],
            summary_caption(averaged, "secondary", "generators and replicates"),
            "tab:agent_extra",
        )
    )
    (args.tex_dir / "agent_libraries.tex").write_text(
        library_table(
            results,
            runs,
            lite,
            averaged,
            "generators and replicates",
        )
    )
    # Each generator on its own: all metrics and per library (appendix).
    for generator, own in runs.groupby("generator", sort=False):
        own_results = {m.key: compare(own, m) for m in MAIN + EXTRA}
        suffix = macro_names[generator].lower()
        (args.tex_dir / f"agent_all_{suffix}.tex").write_text(
            summary_table(
                [(m, own_results[m.key]) for m in MAIN + EXTRA],
                summary_caption(f"{tex_names[generator]} only", "all", "replicates"),
                f"tab:agent_all_{suffix}",
            )
        )
        (args.tex_dir / f"agent_libraries_{suffix}.tex").write_text(
            library_table(
                own_results,
                own,
                lite,
                f"{tex_names[generator]} only",
                "replicates",
                f"tab:agent_libraries_{suffix}",
            )
        )
    moderated = [
        next(m for m in MAIN + EXTRA if m.key == key)
        for key in ("module_f1", "module_recall", "all_module_f1", "held_out_pass_rate")
    ]
    mod, median, mod_frame = moderation(results, runs, moderated)
    cov_models = coverage_models(coverage_units(runs))
    cov = library_moderators(runs)["cov"]
    (args.tex_dir / "agent_contract.tex").write_text(
        moderation_table(mod, cov_models, moderated)
    )
    mod_frame.to_csv(args.contract_csv, index=False)
    library_frame(
        runs, [m for m in MAIN if m.key in ("module_f1", "held_out_pass_rate")]
    ).to_csv(args.libraries_csv, index=False)
    primary = [next(m for m in MAIN if m.key == key) for key in PRIMARY]
    generators, gen = by_generator(runs, primary)
    (args.tex_dir / "agent_generators.tex").write_text(
        generator_table(generators, gen, primary, runs, tex_names)
    )
    (args.tex_dir / "agent_numbers.tex").write_text(
        number_macros(results, runs, total, adjusted)
        + moderation_macros(mod, moderated, median, runs)
        + generator_macros(
            generators,
            gen,
            primary,
            macro_names,
        )
        + coverage_macros(cov_models)
    )

    print("\nGain ~ documentation coverage / length (library x generator):")
    for key, r in cov_models.items():
        c, ln, both = r["cov"], r["len"], r["both"]
        print(
            f"  {key}: n={r['n']} libs={r['libs']}"
            f" cov b={c['z_doc_cov'][0]:.3f} p={c['z_doc_cov'][1]:.3f};"
            f" len b={ln['z_length'][0]:.3f} p={ln['z_length'][1]:.3f};"
            f" both: cov b={both['z_doc_cov'][0]:.3f} p={both['z_doc_cov'][1]:.3f},"
            f" len b={both['z_length'][0]:.3f} p={both['z_length'][1]:.3f}"
        )

    pd.set_option("display.width", 200)
    table = pd.DataFrame(
        {
            m.key: {k: v for k, v in results[m.key].items() if k != "per_library"}
            for m in MAIN + EXTRA
        }
    ).T
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print("Holm (primary):", {k: round(v, 4) for k, v in adjusted.items()})
    print(f"\nModeration by contract coverage (median {median:.2f}):")
    for key, r in mod.items():
        halves = ", ".join(
            f"{h} n={v['n']} diff={v['diff']:.3f} p={v['p']:.3f}"
            for h, v in r["halves"].items()
        )
        print(
            f"  {key}: n={r['n']} rho={r['rho']:.3f} (p={r['p']:.3f});"
            f" doc rho={r['doc_rho']:.3f} (p={r['doc_p']:.3f}, n={r['doc_n']});"
            f" {halves}; interaction p={r['p_interaction']:.3f}"
        )
    print(
        mod_frame[mod_frame["outcome"] == "module_f1"][
            ["name", "cov", "doc_cov", "intent", "full", "diff"]
        ]
        .sort_values("cov")
        .round(3)
        .to_string(index=False)
    )
    print("\nPer generator (Delta, Wilcoxon p; generator p; interaction p):")
    for key, g in gen.items():
        per = ", ".join(
            f"{name} {r['diff']:.3f} (p={r['p_wilcoxon']:.3f})"
            for name, r in g["per"].items()
        )
        print(
            f"  {key}: {per}; generator p={g['p_generator']:.3f};"
            f" interaction p={g['p_interaction']:.3f}"
        )
    print(
        f"Wrote {args.tex_dir}/agent_{{main,extra,libraries,numbers,contract,"
        "generators}.tex"
    )


if __name__ == "__main__":
    main()
