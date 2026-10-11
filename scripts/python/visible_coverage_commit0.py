#!/usr/bin/env python3
"""Does the gain of Full depend on what the visible tests already reveal?

Exploratory. Per held-out test: its *footprint*, the library API names it
mentions (identifiers in the test function, decorators included, that the
reference implementation defines as a module, class, function, method or
class attribute; builtins and container methods excluded), and how much of
that footprint the visible tests already show (any identifier in the test
files the agent sees, helpers and conftest included):

  novelty   share of the footprint no visible test file mentions
  novel     novelty > 0 (at least one API name only the held-out test uses)
  overlap   max Jaccard similarity of the footprint to one visible test's

Tests without an API footprint are left out. Outcomes are per test, per
valid run (analyze_commit0.split_valid); per test and condition the pass
rate is averaged over replicates and rounds (generators), as in the paper.

Outputs:
  - per library and stratum (covered / novel), the pass rate per condition
    and Delta = Full - Intent; paired Wilcoxon over libraries of
    Delta(novel) - Delta(covered);
  - per test, Delta ~ novel (resp. novelty, overlap) + (1 | library);
  - per library, Spearman of the overall held-out Delta with the share of
    novel held-out tests.

Usage:
    uv run --with statsmodels scripts/python/visible_coverage_commit0.py \\
        --round Gemma data/out/benchmark/commit0-gemma \\
        --round Qwen data/out/benchmark/commit0 --csv tests.csv
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import tokenize
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_commit0 import split_valid
from scipy import stats

from reverse_se.benchmark import commit0

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("intent_only", "req_arch")
STOP = (
    set(dir(builtins))
    | {n for t in (dict, list, str, set, bytes, object, int, float) for n in dir(t)}
    | {"self", "cls", "args", "kwargs", "pytest", "mock", "raises", "fixture"}
)


def _parse(path: Path) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, RecursionError, OSError):
        return None


def _identifiers(node: ast.AST) -> set[str]:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg:
            names.add(n.arg)
        elif isinstance(n, ast.alias):
            names.update(n.name.split("."))
            if n.asname:
                names.add(n.asname)
    return names


def _token_names(path: Path) -> set[str]:
    names = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for tok in tokenize.generate_tokens(f.readline):
                if tok.type == tokenize.NAME:
                    names.add(tok.string)
    except (tokenize.TokenError, SyntaxError):
        pass  # keep the names read up to the error
    return names


def api_vocabulary(repo: Path, instance: dict) -> set[str]:
    """Public names the reference implementation defines."""
    vocab = set()
    for path in repo.rglob("*.py"):
        rel = path.relative_to(repo).as_posix()
        if rel.split("/")[0] in (".venv", ".git") or not commit0.is_impl_file(
            rel, instance
        ):
            continue
        vocab.add(path.stem if path.name != "__init__.py" else path.parent.name)
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                vocab.add(node.name)
            if isinstance(node, (ast.ClassDef, ast.Module)):
                for stmt in node.body:
                    for t in getattr(stmt, "targets", [getattr(stmt, "target", None)]):
                        if isinstance(t, ast.Name):
                            vocab.add(t.id)
    return {v for v in vocab if not v.startswith("_") and v not in STOP and len(v) > 2}


def _find_def(tree: ast.Module, path: list[str]) -> ast.AST | None:
    """The function at Class::...::func in a test module."""
    body = tree.body
    for i, name in enumerate(path):
        last = i == len(path) - 1
        for node in body:
            if last and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name == name:
                    return node
            elif not last and isinstance(node, ast.ClassDef) and node.name == name:
                body = node.body
                break
        else:
            if not last:
                return None
    # Inherited test: defined on a base class in the same module.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == path[-1]
        ):
            return node
    return None


class LibraryTests:
    def __init__(self, out: Path, instance: dict, scaffold: str):
        sdir = out / "scaffolds" / scaffold / instance["name"]
        ref = ROOT / "data/repository" / instance["project_id"]
        self.vocab = api_vocabulary(ref, instance)
        test_ids = json.loads((sdir / "test_ids.json").read_text())
        self.held_out = json.loads((sdir / "held_out.json").read_text())
        held = [t for t in test_ids if commit0.is_held_out(t, self.held_out)]
        visible = [t for t in test_ids if not commit0.is_held_out(t, self.held_out)]
        # What the agent sees: every test-tree file of the scaffold, read as
        # tokens -- the split can leave a class with no body, which no
        # longer parses (more-itertools).
        seen = set()
        for path in (sdir / "repo").rglob("*.py"):
            rel = path.relative_to(sdir / "repo").as_posix()
            if commit0.is_test_file(rel, instance):
                seen |= _token_names(path)
        self.seen = seen & self.vocab
        trees: dict[str, ast.Module | None] = {}

        def footprint(tid: str) -> set[str] | None:
            rel, *parts = tid.split("::")
            parts[-1] = parts[-1].split("[")[0]
            if rel not in trees:
                trees[rel] = _parse(ref / rel)
            node = trees[rel] and _find_def(trees[rel], parts)
            return None if node is None else _identifiers(node) & self.vocab

        visible_fp = [fp for t in visible if (fp := footprint(t))]
        self.rows = []
        for tid in held:
            fp = footprint(tid)
            row = {"test": tid, "found": fp is not None, "n_api": len(fp or ())}
            if fp:
                row["novelty"] = len(fp - self.seen) / len(fp)
                row["overlap"] = max(
                    (len(fp & v) / len(fp | v) for v in visible_fp), default=0.0
                )
            self.rows.append(row)


def outcomes(run_dir: Path) -> dict[str, bool]:
    report = run_dir / "grade" / "report.json"
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        return {
            t["nodeid"]: t["call"]["outcome"] == "passed"
            for t in data.get("tests", [])
            if t.get("call")
        }
    return {
        k: v == "passed"
        for k, v in commit0._salvage_outcomes(
            report.with_name("test_output.txt")
        ).items()
    }


def collect(rounds: list[tuple[str, Path]], scaffold: str) -> pd.DataFrame:
    instances = {
        i["name"]: i
        for i in commit0.load_instances(
            ROOT / "data/raw/commit0/commit0_instances.jsonl"
        )
    }
    frames = []
    for label, out in rounds:
        runs = pd.DataFrame(
            [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
        )
        valid, _ = split_valid(runs[runs["scaffold"] == scaffold])
        tests = {
            n: LibraryTests(out, instances[n], scaffold)
            for n in sorted(set(valid.name))
        }
        for r in valid.itertuples():
            run_dir = (
                out
                / "runs"
                / r.model.replace("/", "__")
                / r.scaffold
                / r.name
                / r.condition
                / f"{int(r.replicate):02d}"
            )
            passed = outcomes(run_dir)
            for row in tests[r.name].rows:
                frames.append(
                    {
                        **row,
                        "round": label,
                        "name": r.name,
                        "condition": r.condition,
                        "replicate": int(r.replicate),
                        "passed": passed.get(row["test"], False),
                    }
                )
    return pd.DataFrame(frames)


def per_test(long: pd.DataFrame) -> pd.DataFrame:
    """Pass rate per test and condition: mean over replicates per round, then
    over rounds; Delta = Full - Intent."""
    keys = ["name", "test", "novelty", "overlap", "n_api"]
    rate = (
        long.groupby([*keys, "round", "condition"])["passed"]
        .mean()
        .groupby([*keys, "condition"])
        .mean()
        .unstack("condition")
        .dropna()
        .reset_index()
    )
    rate["delta"] = rate["req_arch"] - rate["intent_only"]
    rate["novel"] = (rate["novelty"] > 0).astype(int)
    return rate


def mixed(frame: pd.DataFrame, predictor: str) -> tuple[float, float, float]:
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(f"delta ~ {predictor}", frame, groups=frame["name"]).fit(
            reml=True
        )
    ci = fit.conf_int().loc[predictor]
    return fit.params[predictor], fit.pvalues[predictor], (ci[0], ci[1])


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--round", nargs=2, action="append", metavar=("LABEL", "OUT"), required=True
    )
    p.add_argument("--scaffold", default="contract")
    p.add_argument("--csv", type=Path, default=None, help="Write per-test table")
    args = p.parse_args()
    long = collect([(lab, Path(o)) for lab, o in args.round], args.scaffold)

    tests = long.drop_duplicates(["name", "test"])
    print(
        f"Held-out tests: {len(tests)}; located in the reference: "
        f"{tests.found.sum()}; with an API footprint: {(tests.n_api > 0).sum()}"
    )
    t = per_test(long[long.n_api > 0])
    if args.csv:
        t.to_csv(args.csv, index=False)
    print(
        f"Novel (>=1 API name no visible test file mentions): {t.novel.sum()} "
        f"of {len(t)} ({100 * t.novel.mean():.0f}%); median novelty "
        f"{t.novelty.median():.2f}, median overlap {t.overlap.median():.2f}"
    )

    print("\nPass rate by stratum (tests pooled, %):")
    print(
        (100 * t.groupby("novel")[["intent_only", "req_arch", "delta"]].mean())
        .rename(index={0: "covered", 1: "novel"})
        .round(1)
        .to_string()
    )

    lib = (
        t.groupby(["name", "novel"])[["intent_only", "req_arch", "delta"]]
        .mean()
        .unstack("novel")
    )
    both = lib.dropna()
    gap = both[("delta", 1)] - both[("delta", 0)]
    w = stats.wilcoxon(gap[gap.abs() > 1e-9]) if (gap.abs() > 1e-9).sum() else None
    print(
        f"\nPer library ({len(both)} with both strata): mean Delta covered "
        f"{100 * both[('delta', 0)].mean():.1f} pp, novel "
        f"{100 * both[('delta', 1)].mean():.1f} pp; novel - covered "
        f"{100 * gap.mean():.1f} pp, larger on novel in {(gap > 1e-9).sum()}, "
        f"smaller in {(gap < -1e-9).sum()}; Wilcoxon p = "
        f"{w.pvalue if w else float('nan'):.3f}"
    )
    rng = np.random.default_rng(0)
    boot = [rng.choice(gap.values, len(gap)).mean() for _ in range(10_000)]
    print(
        f"  bootstrap 95% CI of novel - covered: "
        f"[{100 * np.percentile(boot, 2.5):.1f}, {100 * np.percentile(boot, 97.5):.1f}] pp"
    )

    print("\nPer test, Delta ~ predictor + (1 | library):")
    for pred in ("novel", "novelty", "overlap"):
        b, pv, (lo, hi) = mixed(t, pred)
        print(
            f"  {pred:8s} beta = {100 * b:6.2f} pp  [{100 * lo:.2f}, {100 * hi:.2f}]"
            f"  p = {pv:.3f}"
        )

    share = t.groupby("name").agg(novel=("novel", "mean"), delta=("delta", "mean"))
    rho, pr = stats.spearmanr(share.novel, share.delta)
    print(
        f"\nPer library: share of novel held-out tests "
        f"{100 * share.novel.min():.0f}-{100 * share.novel.max():.0f}% "
        f"(median {100 * share.novel.median():.0f}%); Spearman with Delta "
        f"rho = {rho:.2f}, p = {pr:.3f}"
    )
    pd.set_option("display.width", 200)
    print("\nPer library (%):")
    view = (100 * lib).round(1)
    view.columns = [f"{c}_{'novel' if n else 'cov'}" for c, n in view.columns]
    view["n_tests"] = t.groupby("name").size()
    view["novel_share"] = (100 * share.novel).round(0)
    print(view.to_string())


if __name__ == "__main__":
    main()
