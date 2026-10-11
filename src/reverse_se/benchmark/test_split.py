"""Split each test file's test functions into a visible subset (left in the
agent's scaffold) and a held-out majority (used only by the independent
evaluator, never shown to the agent).

Why: with the *complete* test suite visible, the tests are an almost fully
unambiguous executable spec -- exact signatures, exceptions, data shapes --
which swamps whatever additional signal a prose requirements/architecture
document could add, and makes an A (intent-only) vs B (intent+requirements+
architecture) comparison largely moot (confirmed empirically: an agent
given the full suite reconstructs the design straight from it). SWE-bench
handles this by hiding its grading tests (FAIL_TO_PASS/PASS_TO_PASS)
entirely; Commit0 (the closest from-scratch-library benchmark) shows the
full suite like our original design did. This module takes the middle
ground both those precedents suggest is reasonable for a from-scratch task
under a real iteration budget: keep enough visible tests that the task
stays tractable (avoids a floor effect from zero test guidance), while
holding out enough that the docs still have real ambiguity to resolve.

Test functions are identified the same way regardless of pytest's
class-based vs flat style: any function/method whose name starts with
"test" (pytest's own collection rule). Selection is a deterministic,
seeded shuffle per file, not "first N" or "last N", so the held-out set
isn't systematically the easy or hard half of the suite.
"""

from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


def _is_test_function(name: str) -> bool:
    return name.startswith("test")


@dataclass
class _SplitCounts:
    total: int = 0
    visible: int = 0


class _TestFilter(ast.NodeTransformer):
    """Drop held-out test function/method defs from the tree in place."""

    def __init__(self, held_out_names: set[str]):
        self.held_out_names = held_out_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if _is_test_function(node.name) and node.name in self.held_out_names:
            return None
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST | None:
        if _is_test_function(node.name) and node.name in self.held_out_names:
            return None
        return node


def _collect_test_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef
        ) and _is_test_function(node.name):
            names.append(node.name)
    return names


def test_function_names(source: str) -> set[str]:
    """Names of the test functions/methods defined in a source file."""
    try:
        return set(_collect_test_names(ast.parse(source)))
    except SyntaxError:
        return set()


def split_test_source(
    source: str,
    visible_fraction: float,
    rng: random.Random,
    weights: dict[str, int] | None = None,
) -> tuple[str, list[str], list[str]] | None:
    """Return (visible_source, visible_names, held_out_names), or None if
    the file couldn't be parsed or has no test functions to split (caller
    keeps the original file as-is in either case).

    With `weights` (test IDs per function name), functions are kept visible
    in shuffled order only while the visible share of *test IDs* stays
    within visible_fraction -- a heavily parametrized function counts for
    its IDs, not as one function (unweighted, the visible share of IDs
    ranged from 0% to 69% across Commit0 libraries).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    names = _collect_test_names(tree)
    if not names:
        return None

    names = sorted(
        set(names)
    )  # dedupe (unlikely but names could repeat across nested scopes)
    rng.shuffle(names)
    if weights is None:
        keep_count = max(1, math.ceil(len(names) * visible_fraction))
        visible_names = set(names[:keep_count])
    else:
        target = visible_fraction * sum(weights.get(n, 0) for n in names)
        visible_names, kept = set(), 0
        for name in names:
            weight = weights.get(name, 0)
            if kept + weight <= target:
                visible_names.add(name)
                kept += weight
        if not visible_names:  # every function alone overshoots: lightest one
            visible_names.add(min(names, key=lambda n: weights.get(n, 0)))
    held_out_names = set(names) - visible_names

    if not held_out_names:
        return None  # too few tests to meaningfully split -- keep all visible

    tree = _TestFilter(held_out_names).visit(tree)
    ast.fix_missing_locations(tree)
    try:
        visible_source = ast.unparse(tree)
    except Exception:
        return None

    return visible_source, sorted(visible_names), sorted(held_out_names)


@dataclass
class TestSplitResult:
    split_files: list[str] = field(default_factory=list)
    unsplit_files: list[str] = field(default_factory=list)
    visible_test_count: int = 0
    held_out_test_count: int = 0
    per_file: dict[str, dict] = field(default_factory=dict)


def split_tests_in_repo(
    repo_path: Path,
    held_out_dir: Path,
    test_paths: list[str],
    visible_fraction: float,
    seed: int,
    weights: dict[str, dict[str, int]] | None = None,
) -> TestSplitResult:
    """For each relative test_paths entry under repo_path: back up the
    original file to held_out_dir/<relpath>, then overwrite repo_path/
    <relpath> with only its visible-subset test functions.

    held_out_dir always gets the *original, complete* file (not just the
    removed functions) -- restoring it at grading time is then a plain
    overwrite, not a merge.
    """
    result = TestSplitResult()
    rng = random.Random(seed)

    for rel in test_paths:
        path = repo_path / rel
        if path.suffix != ".py" or not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        split = split_test_source(
            source, visible_fraction, rng, weights.get(rel) if weights else None
        )
        if split is None:
            result.unsplit_files.append(rel)
            continue

        visible_source, visible_names, held_out_names = split
        backup_path = held_out_dir / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(source, encoding="utf-8")
        path.write_text(visible_source, encoding="utf-8")

        result.split_files.append(rel)
        result.visible_test_count += len(visible_names)
        result.held_out_test_count += len(held_out_names)
        result.per_file[rel] = {"visible": visible_names, "held_out": held_out_names}

    return result
