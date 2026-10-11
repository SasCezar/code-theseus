"""Statistics over generated documents, per (model, task), for the paper's
"Generated Documents Stats" results: lengths (chars, words, tokens),
outputs cut off at the token limit, and -- given --mermaid, the per-diagram
output of scripts/bash/check_mermaid.sh -- how many C4 diagrams render,
and -- given --dataset -- the share of each repository's code units that its
requirements and architecture documents name (see doc_coverage.py).

Walks the same llm_results/{model}/{context}/{task}/*.md layout as the
judge (see judge/discovery.py), so it stays correct across sample sizes
(100-stratified now, the full 1000 later) without any path changes.

Diagram failures are grouped by the renderer's error message: parse errors
(invalid syntax), undefined elements (a relationship to an element the
diagram never declares; `undefined_elsewhere` is the share of those declared
in another diagram of the same document), and other.

With --tex_dir (needs --labels), also writes the paper's table
(doc_stats.tex) and \\newcommand macros for the numbers in the text
(doc_numbers.tex).

Usage:
    uv run python -m reverse_se.evaluation.doc_stats \\
        --results_dir data/out/llm_results \\
        --context_budget ctx95 \\
        --output data/out/doc_checks/doc_length_stats.csv \\
        [--dataset data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000_stratified.jsonl] \\
        [--mermaid data/out/doc_checks/mermaid_check.tsv] \\
        [--labels hosted_vllm--gemma-4-31b='\\mGemma=Gemma' \\
                  hosted_vllm--qwen-3.5-122b='\\mQwen=Qwen' \\
                  hosted_vllm--gpt-oss-120b='\\mGpto=Gpto' --tex_dir paper/tables]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import tiktoken

from reverse_se.evaluation.doc_coverage import unit_coverage
from reverse_se.evaluation.judge.discovery import discover_results

# A stable, model-agnostic tokenizer just for a consistent length proxy
# across models — not meant to match any specific model's own tokenizer.
ENCODING = tiktoken.get_encoding("cl100k_base")
TASK_NAMES = {"intent": "Intent", "req": "Requirements", "c4": "Architecture"}
_REL = re.compile(r'C4 rel "([^"]+)" -> "([^"]+)"')
_BLOCK = re.compile(r"```mermaid\n(.*?)```", re.S)


def _finish_reason(md_path: Path) -> str | None:
    meta = md_path.with_suffix(".meta.json")
    if not meta.is_file():
        return None
    return json.loads(meta.read_text(encoding="utf-8")).get("finish_reason")


def _error_type(message: str) -> str:
    if re.search(r"Parse error|Lexical error|No diagram type", message):
        return "parse"
    if "unknown shape" in message:
        return "undefined"
    return "other"


def _declared_elsewhere(row, results_dir: Path) -> bool:
    """Whether the undeclared end(s) of a failing relationship are declared
    (`Kind(alias, ...`) in another diagram of the same document."""
    match = _REL.search(row.error)
    if not match:
        return False
    path = results_dir / row.model / row.context / row.task / f"{row.repo_id}.md"
    blocks = _BLOCK.findall(path.read_text(encoding="utf-8", errors="replace"))
    index = int(row.diagram_idx) - 1
    own = blocks[index] if index < len(blocks) else ""

    def declares(block: str, name: str) -> bool:
        return re.search(rf"\w+\(\s*{re.escape(name)}\s*,", block) is not None

    missing = [n for n in match.groups() if not declares(own, n)]
    others = [b for i, b in enumerate(blocks) if i != index]
    return bool(missing) and all(any(declares(b, n) for b in others) for n in missing)


def quartiles(values: pd.Series, name: str) -> dict:
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return {f"{name}_q1": q1, f"{name}_median": median, f"{name}_q3": q3}


def median_iqr(d, name: str, fmt: str = ",.0f") -> str:
    """`median [q1--q3]` table cell from quartiles() fields."""
    q1, median, q3 = (d[f"{name}_{k}"] for k in ("q1", "median", "q3"))
    return f"{median:{fmt}} {{\\scriptsize [{q1:{fmt}}--{q3:{fmt}}]}}"


def diagram_stats(
    mermaid_tsv: Path, docs: pd.DataFrame, results_dir: Path
) -> pd.DataFrame:
    """Per model: diagrams per C4 document, share that render, share of
    documents whose diagrams all render, and counts of
    failures by type."""
    diagrams = pd.read_csv(mermaid_tsv, sep="\t", dtype=str, keep_default_na=False)
    c4 = docs[docs["task"] == "c4"]
    diagrams = diagrams.merge(
        c4[["model", "context_budget", "repo_id"]].rename(
            columns={"context_budget": "context"}
        )
    )
    diagrams["error_type"] = diagrams["error"].map(_error_type)
    rows = {}
    for model, own in diagrams.groupby("model"):
        n_docs = int((c4["model"] == model).sum())
        by_doc = own.groupby("repo_id")["status"]
        all_ok = by_doc.apply(lambda s: (s == "ok").all())
        # Diagrams per document (0 for documents without any).
        counts = by_doc.size().reindex(
            c4[c4["model"] == model]["repo_id"], fill_value=0
        )
        failed = own[own["status"] == "fail"]
        undefined = failed[failed["error_type"] == "undefined"]
        rows[model] = {
            "diagrams": len(own),
            "per_doc": len(own) / n_docs,
            **quartiles(counts, "per_doc"),
            "valid_pct": 100 * (own["status"] == "ok").mean(),
            "docs_all_valid_pct": 100 * all_ok.sum() / n_docs,
            "failed": len(failed),
            **{
                f"fail_{kind}": int((failed["error_type"] == kind).sum())
                for kind in ("parse", "undefined", "other")
            },
            "undefined_elsewhere": sum(
                _declared_elsewhere(r, results_dir) for r in undefined.itertuples()
            ),
        }
    return pd.DataFrame(rows).T


def write_tex(
    docs: pd.DataFrame,
    diagrams: pd.DataFrame,
    coverage: pd.DataFrame,
    labels: dict,
    tex_dir: Path,
) -> None:
    def cmd(name: str, value) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"

    macros = ["% Generated by reverse_se.evaluation.doc_stats -- do not edit by hand."]
    lines = [
        "\\begin{table*}[htbp!]",
        "\\centering",
        "\\small",
        "\\caption{Generated documents, medians [IQR]. Units named: share of a"
        " repository's code units (modules, files or types; tests excluded) that"
        " its requirements or architecture document names. Diagrams: Mermaid C4"
        " diagrams per architecture document, share that render, and share of"
        " documents whose diagrams all render.}",
        "\\label{tab:doc_stats}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{3}{c}{\\textbf{Length (words)}} & \\textbf{Units}"
        " & \\multicolumn{3}{c}{\\textbf{C4 Mermaid diagrams}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){6-8}",
        "\\textbf{Model} & \\textbf{Intent} & \\textbf{Requirements}"
        " & \\textbf{Architecture} & \\textbf{Named (\\%)}"
        " & \\textbf{Per doc.} & \\textbf{Valid (\\%)}"
        " & \\textbf{Docs all valid (\\%)} \\\\",
        "\\midrule",
    ]
    named = {
        model: quartiles(own["units_named_pct"], "named")
        for model, own in coverage.groupby("model")
    }
    for model, (label, prefix) in labels.items():
        own = docs[docs["model"] == model]
        cells = [label]
        for task, name in TASK_NAMES.items():
            words = quartiles(own[own["task"] == task]["n_words"], "words")
            cells.append(median_iqr(words, "words"))
            macros.append(cmd(f"{prefix}{name}Words", f"{words['words_median']:,.0f}"))
        cells.append(median_iqr(named[model], "named"))
        macros.append(cmd(f"{prefix}UnitsNamed", f"{named[model]['named_median']:.0f}"))
        d = diagrams.loc[model]
        cells.append(median_iqr(d, "per_doc"))
        cells += [f"{d['valid_pct']:.1f}", f"{d['docs_all_valid_pct']:.1f}"]
        lines.append(" & ".join(cells) + " \\\\")
        macros += [
            cmd(f"{prefix}Docs", f"{int((own['task'] == 'c4').sum()):,}"),
            cmd(f"{prefix}Truncated", int((own["finish_reason"] == "length").sum())),
            cmd(f"{prefix}DiagramsPerDoc", f"{d['per_doc_median']:.0f}"),
            cmd(f"{prefix}DiagramsValid", f"{d['valid_pct']:.1f}"),
            cmd(f"{prefix}DocsValid", f"{d['docs_all_valid_pct']:.1f}"),
        ]
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    ours = diagrams.loc[list(labels)]
    failed = ours["failed"].sum()
    macros += [
        cmd("DocsTotal", f"{len(docs):,}"),
        cmd("DocsTruncated", int((docs["finish_reason"] == "length").sum())),
        cmd("DiagramsFailed", f"{int(failed):,}"),
        *(
            cmd(
                f"Fail{kind.capitalize()}",
                f"{100 * ours[f'fail_{kind}'].sum() / failed:.0f}",
            )
            for kind in ("parse", "undefined", "other")
        ),
        cmd(
            "UndefinedElsewhere",
            f"{100 * ours['undefined_elsewhere'].sum() / ours['fail_undefined'].sum():.0f}",
        ),
    ]
    ours = coverage[coverage["model"].isin(labels)]
    macros.append(cmd("UnitsReposNamed", f"{ours['repo_id'].nunique():,}"))
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / "doc_stats.tex").write_text("\n".join(lines))
    (tex_dir / "doc_numbers.tex").write_text("\n".join(macros) + "\n")
    print(f"\nWrote {tex_dir}/doc_stats.tex and doc_numbers.tex")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--context_budget",
        default=None,
        help="Filter to this context budget dir (e.g. ctx95)",
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=Path("doc_length_stats.csv"))
    parser.add_argument(
        "--mermaid", type=Path, default=None, help="check_mermaid.sh output TSV"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[],
        metavar="MODEL=LABEL[=MACRO]",
        help="Models to keep, with their name in the paper (and a letters-only "
        "macro prefix when the label isn't one)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Keep only documents of the repos in this dataset JSONL (e.g. the "
        "evaluation sample); other docs in the same folders, such as the "
        "Commit0 benchmark's, are left out",
    )
    parser.add_argument("--tex_dir", type=Path, default=None)
    args = parser.parse_args()
    repos = None
    if args.dataset:
        with open(args.dataset, encoding="utf-8") as f:
            repos = {json.loads(line)["repo_id"] for line in f if line.strip()}
    labels = {}
    for spec in args.labels:
        model, label, *prefix = spec.split("=")
        labels[model] = (label, prefix[0] if prefix else label)

    rows = []
    for model, ctx, task, repo_id, path in discover_results(
        args.results_dir, args.task, args.model
    ):
        if args.context_budget and ctx != args.context_budget:
            continue
        if labels and model not in labels:
            continue
        if repos is not None and repo_id not in repos:
            continue
        text = path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "model": model,
                "context_budget": ctx,
                "task": task,
                "repo_id": repo_id,
                "n_chars": len(text),
                "n_words": len(text.split()),
                "n_tokens": len(ENCODING.encode(text)),
                "finish_reason": _finish_reason(path),
            }
        )

    if not rows:
        raise SystemExit(f"No results found under {args.results_dir}")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    summary = df.groupby(["model", "task"])[["n_chars", "n_words", "n_tokens"]].agg(
        [
            "mean",
            "std",
            "median",
            lambda s: s.quantile(0.25),
            lambda s: s.quantile(0.75),
        ]
    )
    summary = summary.rename(columns={"<lambda_0>": "q1", "<lambda_1>": "q3"}).round(1)
    print(f"\nDoc length stats ({len(df)} docs), per (model, task):")
    print(summary.to_string())

    overall = (
        df.groupby("model")[["n_chars", "n_words", "n_tokens"]]
        .agg(["mean", "std"])
        .round(1)
    )
    print("\nPer-model (all tasks pooled):")
    print(overall.to_string())
    truncated = df[df["finish_reason"] == "length"].groupby(["model", "task"]).size()
    print("\nCut off at the token limit:", truncated.to_dict() or "none")

    print(f"\nWrote per-doc length stats -> {args.output}")

    coverage = None
    if args.dataset:
        coverage = unit_coverage(
            args.dataset,
            args.results_dir,
            args.context_budget or "ctx95",
            sorted(df["model"].unique()),
        )
        coverage.to_csv(args.output.with_name("doc_unit_coverage.csv"), index=False)
        print("\nCode units named in the requirements and architecture docs (%):")
        print(
            coverage.groupby("model")["units_named_pct"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .round(1)
            .to_string()
        )

    if args.mermaid:
        diagrams = diagram_stats(args.mermaid, df, args.results_dir)
        print("\nMermaid C4 diagrams, per model:")
        print(diagrams.round(1).to_string())
        if args.tex_dir:
            if not labels or coverage is None:
                raise SystemExit("--tex_dir needs --labels and --dataset")
            write_tex(df, diagrams, coverage, labels, args.tex_dir)


if __name__ == "__main__":
    main()
