#!/usr/bin/env python3
"""Rank candidate projects by structural ambiguity --
how much room there is for two competent-but-different decompositions of
the same intent, and how much the test suite actually exercises structure
(cross-module calls) rather than a single module's internal values.

Motivation: an architecture doc can only pay off on a project where "which
components exist and how they talk to each other" is genuinely
underdetermined by intent alone, and where the (held-out) tests would
notice if the agent picked a different decomposition. A tiny project with
one obvious folder layout, or a project whose tests are mostly single-module
value assertions, gives an architecture doc nothing to contribute -- see the
d-Rickyy-b|Python-BlackJackBot investigation this script's metrics are
meant to generalize (its 4 req_arch failures were all held-out, single-
module, incidental-value assertions -- e.g. an exact emoji string -- not
cross-module interface mismatches).

Python-only (AST-based); other languages are reported with size-only
fallback columns and flagged so they aren't silently ranked on missing data.

Metrics, computed against the exact baseline commit already checked out at
data/repository/<project_id> (for Commit0: the reference commit,
checked out by scripts/bash/prefetch_commit0.sh):
  - n_modules: immediate subpackages/top-level source files under the
    project's source root -- a coarse proxy for "how many pieces are there
    to wire together".
  - n_classes / n_public_defs: AST class/def counts (excludes test files).
  - intra_project_import_edges: distinct (module_a -> module_b) edges from
    parsing `import`/`from` statements and resolving them against the
    project's own top-level module names only (stdlib/third-party imports
    are not edges). More edges relative to n_modules means more real
    coupling for an architecture doc to usefully describe.
  - cross_module_test_fraction: of the project's test files (visible +
    held-out, i.e. everything under a detected test path), the fraction
    that import from more than one distinct project-internal top-level
    module -- a proxy for "this test exercises an integration/interface,
    not just one module's internal values".

No single composite score is computed on purpose (the weighting between
these would be an arbitrary judgment call) -- this prints the raw table,
sorted by cross_module_test_fraction then import edge density, and leaves
picking a threshold to a human.

Usage:
    uv run scripts/python/structural_ambiguity.py
    uv run scripts/python/structural_ambiguity.py --benchmark-input <records.jsonl>

Input records need `project_id` (checkout at data/repository/<project_id>);
`language` defaults to Python and `source_loc` to 0 -- the defaults the
Commit0 instance list (data/raw/commit0/commit0_instances.jsonl) relies on.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

from utils import ROOT_DIR

from reverse_se.benchmark.test_paths import is_test_path

DEFAULT_BENCHMARK_INPUT = ROOT_DIR / "data/raw/commit0/commit0_instances.jsonl"
DEFAULT_REPOSITORY_BASE = ROOT_DIR / "data/repository"

_EXCLUDE_DIR_NAMES = {
    ".git",
    "venv",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    "node_modules",
}


@dataclass
class ProjectMetrics:
    project_id: str
    language: str
    source_loc: int
    n_modules: int = 0
    n_classes: int = 0
    n_public_defs: int = 0
    intra_project_import_edges: int = 0
    n_test_files: int = 0
    cross_module_test_fraction: float | None = None
    note: str = ""


def _iter_source_files(repo_root: Path, extensions: frozenset[str]) -> list[Path]:
    files = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix not in extensions:
            continue
        if any(
            part in _EXCLUDE_DIR_NAMES
            for part in path.relative_to(repo_root).parts[:-1]
        ):
            continue
        files.append(path)
    return files


def _component_of(rel_path: Path) -> str:
    # One node per *directory* (subpackage), not per top-level package --
    # "blackjack/game/deck.py" -> "blackjack/game", so imports between
    # sibling subpackages of the same top-level package (the common
    # single-top-package layout: green/, physt/, ...) still register as
    # edges instead of all collapsing into one "module". A root-level
    # single-file module (e.g. "setup.py") is its own component, keyed by
    # its stem so it doesn't collide with an unrelated "." node.
    parent = rel_path.parent
    return parent.as_posix() if parent != Path(".") else rel_path.stem


def _resolve_import(dotted: str, known_components: frozenset[str]) -> str | None:
    # Longest-prefix match: the dotted name may point at a package
    # (component itself), or at a specific module/symbol *within* one
    # (trailing segments that aren't themselves directories) -- trying
    # progressively shorter prefixes handles both without needing to know
    # which case it is ahead of time.
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        candidate = "/".join(parts[:i])
        if candidate in known_components:
            return candidate
    return None


@dataclass
class _ImportVisitor(ast.NodeVisitor):
    known_components: frozenset[str]
    resolved: set[str] = field(default_factory=set)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            target = _resolve_import(alias.name, self.known_components)
            if target:
                self.resolved.add(target)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            return  # relative import -- same-package by construction, not a cross-component edge
        if node.module:
            target = _resolve_import(node.module, self.known_components)
            if target:
                self.resolved.add(target)


def _analyze_python(
    repo_root: Path,
) -> tuple[int, int, int, int, int, float | None, str]:
    py_files = _iter_source_files(repo_root, frozenset({".py"}))
    if not py_files:
        return 0, 0, 0, 0, 0, None, "no .py files found"

    source_files = [f for f in py_files if not is_test_path(f.relative_to(repo_root))]
    test_files = [f for f in py_files if is_test_path(f.relative_to(repo_root))]

    known_components = frozenset(
        _component_of(f.relative_to(repo_root)) for f in source_files
    )
    n_modules = len(known_components)

    n_classes = 0
    n_public_defs = 0
    edges: set[tuple[str, str]] = set()
    parse_errors = 0

    for f in source_files:
        rel = f.relative_to(repo_root)
        own_component = _component_of(rel)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            parse_errors += 1
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                n_classes += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                not node.name.startswith("_")
                or (node.name.startswith("__") and node.name.endswith("__"))
            ):
                n_public_defs += 1
        visitor = _ImportVisitor(known_components=known_components)
        visitor.visit(tree)
        for target in visitor.resolved:
            if target != own_component:
                edges.add((own_component, target))

    cross_module_test_fraction = None
    if test_files:
        cross_count = 0
        for f in test_files:
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            visitor = _ImportVisitor(known_components=known_components)
            visitor.visit(tree)
            if len(visitor.resolved) > 1:
                cross_count += 1
        cross_module_test_fraction = cross_count / len(test_files)

    note = f"{parse_errors} file(s) failed to parse" if parse_errors else ""
    return (
        n_modules,
        n_classes,
        n_public_defs,
        len(edges),
        len(test_files),
        cross_module_test_fraction,
        note,
    )


def extract_boundary_identifiers(repo_root: Path) -> set[str]:
    """Real class/function names that some *other* component actually
    imports by name somewhere in the project -- the public interface
    surface between components, as opposed to every internal
    implementation detail. This is the identifier set a proper
    architecture doc should name (component boundaries and the interfaces
    they expose are exactly what architecture documentation is for);
    scoring a doc against *all* identifiers (as doc_groundedness.py
    originally did) unfairly penalizes it for not repeating purely
    internal, non-architectural names.
    """
    py_files = _iter_source_files(repo_root, frozenset({".py"}))
    source_files = [f for f in py_files if not is_test_path(f.relative_to(repo_root))]
    known_components = frozenset(
        _component_of(f.relative_to(repo_root)) for f in source_files
    )

    boundary: set[str] = set()
    for f in source_files:
        rel = f.relative_to(repo_root)
        own_component = _component_of(rel)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue

        # Pass 1: `from X import Y` -- Y is directly the cross-component name.
        # Also track `import X[.Y] [as Z]` bindings resolved to a *different*
        # component, so pass 2 can catch the `import X.Y; X.Y.symbol(...)`
        # attribute-access style, which is just as common and would
        # otherwise silently score as "no boundary identifiers" even when
        # real cross-component calls exist.
        binding_to_component: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    target = _resolve_import(alias.name, known_components)
                    if target and target != own_component:
                        binding_to_component[bound] = target
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if not node.module:
                    continue
                target = _resolve_import(node.module, known_components)
                if not target or target == own_component:
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        boundary.add(alias.name)

        # Pass 2: attribute access off a tracked cross-component binding,
        # e.g. `import blackjack.game as game; game.Deck(...)`.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in binding_to_component:
                boundary.add(node.attr)
    return boundary


def analyze_project(
    project_id: str, language: str, source_loc: int, repo_base: Path
) -> ProjectMetrics:
    slug = project_id  # stored as "owner|repo", the data/repository dir name
    repo_root = repo_base / slug
    if not repo_root.is_dir():
        return ProjectMetrics(
            project_id, language, source_loc, note="repository checkout not found"
        )

    if language != "Python":
        return ProjectMetrics(
            project_id,
            language,
            source_loc,
            note="non-Python -- size-only, not AST-analyzed",
        )

    n_modules, n_classes, n_public_defs, edges, n_test_files, cross_frac, note = (
        _analyze_python(repo_root)
    )
    return ProjectMetrics(
        project_id=project_id,
        language=language,
        source_loc=source_loc,
        n_modules=n_modules,
        n_classes=n_classes,
        n_public_defs=n_public_defs,
        intra_project_import_edges=edges,
        n_test_files=n_test_files,
        cross_module_test_fraction=cross_frac,
        note=note,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark-input", type=Path, default=DEFAULT_BENCHMARK_INPUT)
    parser.add_argument("--repository-base", type=Path, default=DEFAULT_REPOSITORY_BASE)
    args = parser.parse_args()

    records = []
    with args.benchmark_input.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    results = [
        analyze_project(
            r["project_id"],
            r.get("language", "Python"),
            r.get("source_loc", 0),
            args.repository_base,
        )
        for r in records
    ]

    def sort_key(m: ProjectMetrics) -> tuple[float, float]:
        frac = (
            m.cross_module_test_fraction
            if m.cross_module_test_fraction is not None
            else -1.0
        )
        density = m.intra_project_import_edges / m.n_modules if m.n_modules else -1.0
        return (frac, density)

    results.sort(key=sort_key, reverse=True)

    header = (
        f"{'project_id':<32} {'lang':<6} {'loc':>6} {'modules':>7} {'classes':>7} "
        f"{'pub_defs':>8} {'edges':>6} {'edge/mod':>8} {'test_files':>10} {'cross_test_frac':>15}  note"
    )
    print(header)
    print("-" * len(header))
    for m in results:
        edge_density = (
            m.intra_project_import_edges / m.n_modules if m.n_modules else float("nan")
        )
        cross_frac_str = (
            f"{m.cross_module_test_fraction:.2f}"
            if m.cross_module_test_fraction is not None
            else "n/a"
        )
        print(
            f"{m.project_id:<32} {m.language:<6} {m.source_loc:>6} {m.n_modules:>7} {m.n_classes:>7} "
            f"{m.n_public_defs:>8} {m.intra_project_import_edges:>6} {edge_density:>8.2f} "
            f"{m.n_test_files:>10} {cross_frac_str:>15}  {m.note}"
        )


if __name__ == "__main__":
    main()
