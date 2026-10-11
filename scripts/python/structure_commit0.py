#!/usr/bin/env python3
"""Structural similarity of agent-built libraries to the reference.

Do agents given requirements + architecture docs (req_arch) build a library
whose *structure* is closer to the real one than agents given the intent
only? Held-out tests can't tell: they check behavior, not where code lives
or how modules depend on each other.

Protocol (fixed before scoring the benchmark's runs)
----------------------------------------------------
Compared: the implementation files (commit0.is_impl_file: under src_dir,
not tests) of each run's final workspace vs. the library's reference
checkout (data/repository/commit-0|<name>, verified to be at
reference_commit). Modules are dotted names (api_scaffold.module_map).

What both conditions get for free is excluded: API_CONTRACT.txt gives
every run the dotted paths the tests import. "Contract modules" are the
reference modules that are a contract name or a prefix of one (e.g.
`babel`, `babel.dates` for `babel.dates.format_date`); "contract symbols"
are the contract names' last components.

Per run:
  module_precision / module_recall / module_f1   (PRIMARY)
      agent vs. reference module names, contract modules excluded from
      both sides. Undefined (NaN; the library is skipped) when the
      contract names every reference module.
  symbol_ari   (PRIMARY)
      Entities: public (no leading underscore) top-level classes and
      functions defined exactly once in both implementations, contract
      symbols excluded. Adjusted Rand index between the two groupings of
      those entities into modules: 1 = same grouping, ~0 = chance. Needs
      >= 2 entities; undefined (NaN) otherwise.
  placement_accuracy
      Share of those entities whose agent module maps to their reference
      module. Mapping: an agent module with the same name as a reference
      module maps to it; otherwise to the reference module it shares the
      most entities with (ties: alphabetical).
  edge_precision / edge_recall / edge_f1
      Internal import edges (module -> module, from import statements,
      relative imports resolved), agent edges mapped through the module
      mapping; self-loops and unmapped ends dropped.
  all_module_f1   (robustness check, added after scoring)
      module F1 with contract modules kept on both sides, so it is
      defined for every library.
  doc_module_recall   (manipulation check, not an outcome)
      Share of non-contract reference modules named in the library's
      req + c4 docs (dotted name, `a/b.py` path, a unique `b.py`, or a C4
      element identifier matching the module name -- the docs rarely name
      files) that the agent created. req_arch should be well above intent_only; if
      not, the agents didn't use the docs.
  agent_modules, agent_loc   (size controls)

Runs: those analyze_commit0.split_valid keeps (audit ok, no launch or
grading error, steps > 0), minus --exclude. Comparison: per library, the
mean over replicates per condition; paired over libraries with both
conditions: req_arch - intent_only, Wilcoxon signed-rank, sign test,
bootstrap 95% CI of the mean.

--anchors prints the scale for each library instead: the reference against
itself (all 1.0), and "everything in one module" (module F1 0, ARI 0).
(Revised after the anchors, before any run was scored: module F1 was 0
instead of undefined when the contract names every module, and the check
matched only file names, which the docs almost never use.)

Usage (on the machine with the runs and reference checkouts):
    uv run scripts/python/structure_commit0.py --anchors
    uv run scripts/python/structure_commit0.py --exclude chardet
        [--out data/out/benchmark/commit0] [--scaffold contract]
        [--docs-dir data/out/llm_results/<model>/ctx95]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_commit0 import split_valid
from scipy import stats

from reverse_se.benchmark import commit0
from reverse_se.benchmark.api_scaffold import module_map

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ("module_f1", "symbol_ari")
SECONDARY = (
    "all_module_f1",
    "placement_accuracy",
    "edge_f1",
    "module_precision",
    "module_recall",
)
CHECK = ("doc_module_recall",)


# ── extraction ────────────────────────────────────────────────────────────────


@dataclass
class Structure:
    modules: set[str] = field(default_factory=set)
    # public top-level class/function name -> defining modules
    symbols: dict[str, set[str]] = field(default_factory=dict)
    edges: set[tuple[str, str]] = field(default_factory=set)
    loc: int = 0


def _parse(path: Path) -> ast.Module | None:
    try:
        with warnings.catch_warnings():  # e.g. invalid escapes in parsed code
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, RecursionError):
        return None


def _imports(tree: ast.Module, module: str, is_package: bool) -> set[str]:
    """Dotted names a module imports (absolute; relative ones resolved).
    For `from x import y` both x.y and x, the caller keeps what exists."""
    package = module if is_package else module.rpartition(".")[0]
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                base = (
                    parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                )
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if not prefix:
                continue
            names.add(prefix)
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def extract(repo: Path, instance: dict) -> Structure:
    impl = sorted(
        p.relative_to(repo).as_posix()
        for p in repo.rglob("*.py")
        if p.relative_to(repo).parts[0] not in (".venv", ".git")
        and commit0.is_impl_file(p.relative_to(repo).as_posix(), instance)
    )
    modules = module_map(repo, impl)
    s = Structure(modules=set(modules))
    for module, rel in modules.items():
        path = repo / rel
        text = path.read_text(encoding="utf-8", errors="replace")
        s.loc += sum(1 for line in text.splitlines() if line.strip())
        tree = _parse(path)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                s.symbols.setdefault(node.name, set()).add(module)
        for name in _imports(tree, module, path.name == "__init__.py"):
            target = _longest_module(name, s.modules)
            if target and target != module:
                s.edges.add((module, target))
    return s


def _longest_module(name: str, modules: set[str]) -> str | None:
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in modules:
            return candidate
    return None


def contract_names(scaffold_repo: Path) -> list[str]:
    path = scaffold_repo / commit0.CONTRACT_FILE
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def contract_parts(names: list[str], ref_modules: set[str]) -> tuple[set, set]:
    modules, symbols = set(), set()
    for name in names:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            if ".".join(parts[:i]) in ref_modules:
                modules.add(".".join(parts[:i]))
        if ".".join(parts[:-1]) in ref_modules or name not in ref_modules:
            symbols.add(parts[-1])
    return modules, symbols


# C4 element identifiers: Container(alias, ...), Component(alias, ...), ...
_C4_ALIAS = re.compile(r"\b(?:Container|Component|System)\w*\(\s*([A-Za-z_]\w*)\s*,")


def doc_modules(text: str, ref: Structure, excluded: set[str]) -> set[str]:
    """Non-contract reference modules the docs name: by dotted name or
    `a/b.py` path, a unique `b.py`, or a C4 element identifier matching the
    module's own name (`localedata`), extending it (`pofile_parser` for
    pofile, `extractor` for extract; 5+ characters) or shortening it."""
    basenames = Counter(m.rpartition(".")[2] for m in ref.modules)
    aliases = {a.lower() for a in _C4_ALIAS.findall(text)}
    found = set()
    for module in ref.modules - excluded:
        last = module.rpartition(".")[2].lower()
        patterns = [re.escape(module), re.escape(module.replace(".", "/") + ".py")]
        if basenames[last] == 1 and len(last) >= 3:
            patterns.append(rf"(?<![\w/.]){re.escape(last)}\.py\b")
        named = any(
            a == last
            or (len(last) >= 5 and a.startswith(last))
            or (len(a) >= 5 and last.startswith(a))
            for a in aliases
        )
        if named or any(re.search(p, text) for p in patterns):
            found.add(module)
    return found


# ── metrics ───────────────────────────────────────────────────────────────────


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def _prf(predicted: set, truth: set) -> tuple[float, float, float]:
    """Precision, recall, F1; all NaN when there's nothing to find (e.g.
    the contract already names every module): the library is skipped for
    that metric rather than scored 0 in both conditions."""
    if not truth:
        return (float("nan"),) * 3
    hit = len(predicted & truth)
    p = hit / len(predicted) if predicted else 0.0
    r = hit / len(truth)
    return p, r, _f1(p, r)


def adjusted_rand(a: list[str], b: list[str]) -> float:
    """ARI between two labelings of the same entities."""
    n = len(a)
    if n < 2:
        return float("nan")

    def pairs(x: float) -> float:
        return x * (x - 1) / 2

    table = Counter(zip(a, b, strict=True))
    index = sum(pairs(v) for v in table.values())
    rows = sum(pairs(v) for v in Counter(a).values())
    cols = sum(pairs(v) for v in Counter(b).values())
    expected = rows * cols / pairs(n)
    maximum = (rows + cols) / 2
    if maximum == expected:  # both groupings trivial (all one / all apart)
        return 1.0 if index == maximum else 0.0
    return (index - expected) / (maximum - expected)


def compare(
    ref: Structure,
    agent: Structure,
    contract_modules: set[str],
    contract_symbols: set[str],
    documented: set[str],
) -> dict:
    p, r, f = _prf(agent.modules - contract_modules, ref.modules - contract_modules)
    _, _, all_f = _prf(agent.modules, ref.modules)
    entities = sorted(
        name
        for name, mods in ref.symbols.items()
        if len(mods) == 1
        and len(agent.symbols.get(name, ())) == 1
        and name not in contract_symbols
    )
    ref_of = {n: next(iter(ref.symbols[n])) for n in entities}
    agent_of = {n: next(iter(agent.symbols[n])) for n in entities}
    mapping = {}
    for module in agent.modules:
        if module in ref.modules:
            mapping[module] = module
            continue
        votes = Counter(ref_of[n] for n in entities if agent_of[n] == module)
        if votes:
            best = max(votes.values())
            mapping[module] = min(m for m, v in votes.items() if v == best)
    placed = [mapping.get(agent_of[n]) == ref_of[n] for n in entities]
    mapped_edges = {
        (mapping[a], mapping[b])
        for a, b in agent.edges
        if a in mapping and b in mapping and mapping[a] != mapping[b]
    }
    ep, er, ef = _prf(mapped_edges, ref.edges)
    _, doc_recall, _ = _prf(agent.modules & documented, documented)
    return {
        "module_precision": p,
        "module_recall": r,
        "module_f1": f,
        "all_module_f1": all_f,
        "symbol_ari": adjusted_rand(
            [agent_of[n] for n in entities], [ref_of[n] for n in entities]
        ),
        "placement_accuracy": float(np.mean(placed)) if placed else float("nan"),
        "shared_entities": len(entities),
        "edge_precision": ep,
        "edge_recall": er,
        "edge_f1": ef,
        "doc_module_recall": doc_recall,
        "documented_modules": len(documented),
        "agent_modules": len(agent.modules),
        "agent_loc": agent.loc,
        "reference_modules": len(ref.modules),
        "contract_modules": len(contract_modules),
    }


# ── driver ────────────────────────────────────────────────────────────────────


def reference_repo(instance: dict) -> Path:
    repo = ROOT / "data/repository" / instance["project_id"]
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    ref = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            f"{instance['reference_commit']}^{{commit}}",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head or head != ref:
        raise SystemExit(
            f"{repo} is not at reference_commit {instance['reference_commit']}"
        )
    return repo


class Library:
    """Reference structure, contract and documented modules of one library."""

    def __init__(self, instance: dict, args):
        self.instance = instance
        self.ref = extract(reference_repo(instance), instance)
        scaffold = args.out / "scaffolds" / args.scaffold / instance["name"] / "repo"
        self.contract_modules, self.contract_symbols = contract_parts(
            contract_names(scaffold), self.ref.modules
        )
        text = "\n".join(
            (args.docs_dir / kind / f"{instance['project_id']}.md").read_text(
                encoding="utf-8", errors="replace"
            )
            for kind in ("req", "c4")
            if (args.docs_dir / kind / f"{instance['project_id']}.md").is_file()
        )
        self.documented = doc_modules(text, self.ref, self.contract_modules)

    def compare(self, agent: Structure) -> dict:
        return compare(
            self.ref,
            agent,
            self.contract_modules,
            self.contract_symbols,
            self.documented,
        )


def anchors(libraries: dict[str, Library]) -> None:
    rows = []
    for name, lib in sorted(libraries.items()):
        ref = lib.ref
        root = min(ref.modules, key=len) if ref.modules else "x"
        flat = Structure(
            modules={root},
            symbols={s: {root} for s in ref.symbols},
            edges=set(),
            loc=ref.loc,
        )
        for label, agent in (("reference", ref), ("one_module", flat)):
            rows.append({"name": name, "anchor": label, **lib.compare(agent)})
    frame = pd.DataFrame(rows)
    cols = ["name", "anchor", *PRIMARY, *SECONDARY, "shared_entities", *CHECK]
    cols += ["documented_modules", "reference_modules"]
    print(frame[cols].round(3).to_string(index=False))


def paired(frame: pd.DataFrame, metric: str, seed: int = 0) -> dict:
    means = frame.groupby(["name", "condition"])[metric].mean().unstack()
    means = means.dropna(subset=[c for c in ("intent_only", "req_arch") if c in means])
    if not {"intent_only", "req_arch"} <= set(means):
        return {"metric": metric, "libraries": 0}
    diffs = (means["req_arch"] - means["intent_only"]).to_numpy()
    out = {
        "metric": metric,
        "libraries": len(diffs),
        "intent_only": means["intent_only"].mean(),
        "req_arch": means["req_arch"].mean(),
        "mean_diff": diffs.mean(),
        "median_diff": float(np.median(diffs)),
        "req_arch_better": int((diffs > 0).sum()),
        "intent_better": int((diffs < 0).sum()),
    }
    if len(diffs):
        boots = np.random.default_rng(seed).choice(diffs, (10_000, len(diffs))).mean(1)
        out["ci95"] = (
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        )
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 1:
        out["wilcoxon_p"] = float(stats.wilcoxon(nonzero).pvalue)
        out["sign_p"] = float(
            stats.binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/out/benchmark/commit0")
    parser.add_argument("--scaffold", default="contract")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--instances",
        type=Path,
        default=ROOT / "data/raw/commit0/commit0_instances.jsonl",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=ROOT / "data/out/llm_results/hosted_vllm--qwen-3.5-122b/ctx95",
    )
    parser.add_argument("--exclude", nargs="+", default=[], metavar="NAME")
    parser.add_argument("--anchors", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="Write per-run metrics")
    args = parser.parse_args()

    instances = {i["name"]: i for i in commit0.load_instances(args.instances)}
    runs = pd.DataFrame(
        [
            json.loads(line)
            for line in (args.out / "results.jsonl").read_text().splitlines()
            if line.strip()
        ]
    )
    runs = runs[(runs["scaffold"] == args.scaffold) & ~runs["name"].isin(args.exclude)]
    if args.model:
        runs = runs[runs["model"] == args.model]
    libraries = {
        name: Library(instances[name], args) for name in sorted(set(runs["name"]))
    }
    if args.anchors:
        anchors(libraries)
        return

    valid, bad = split_valid(runs)
    print(f"{len(runs)} runs, {len(valid)} valid, {len(bad)} excluded")
    rows = []
    for r in valid.itertuples():
        run_dir = (
            args.out
            / "runs"
            / r.model.replace("/", "__")
            / r.scaffold
            / r.name
            / r.condition
            / f"{int(r.replicate):02d}"
        )
        agent = extract(run_dir / "workspace", instances[r.name])
        rows.append(
            {
                "name": r.name,
                "condition": r.condition,
                "replicate": int(r.replicate),
                "held_out_pass_rate": r.held_out_pass_rate,
                **libraries[r.name].compare(agent),
            }
        )
    frame = pd.DataFrame(rows)
    if args.csv:
        frame.to_csv(args.csv, index=False)
    pd.set_option("display.width", 220)
    per_lib = frame.groupby(["name", "condition"])[
        [*PRIMARY, *SECONDARY, *CHECK]
    ].mean()
    print("\nPer library (replicates averaged):")
    print(per_lib.unstack("condition").round(3).to_string())
    print("\nPaired over libraries (req_arch - intent_only):")
    summary = pd.DataFrame([paired(frame, m) for m in (*PRIMARY, *SECONDARY, *CHECK)])
    print(summary.round(3).to_string(index=False))
    print(
        "\nPRIMARY: module_f1, symbol_ari. doc_module_recall is a manipulation "
        "check (did the agents use the docs?), not an outcome."
    )


if __name__ == "__main__":
    sys.exit(main())
