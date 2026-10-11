#!/usr/bin/env python3
"""Check whether generated req_arch (c4_doc) docs actually name the real
project's classes/functions, or just restate intent in more prose without
adding structural (component/interface-level) information.

Motivation: an architecture doc can only cause a measurable behavior
difference vs. intent_only if it actually carries structural content --
concrete component/class names, responsibilities, interfaces -- beyond what
the intent doc already says. This is a cheap, automatable first-pass proxy
for that, not a full semantic grade: it extracts every real class/public-
function identifier from the reference implementation (the data/repository checkout) and checks how many of them are literally
named (word-boundary match) in the c4_doc vs. the intent_doc. A doc that
never names any real class is almost certainly high-level filler; a doc
that names most of them is at least trying to convey real structure.
Paraphrases ("the handler for banned users" instead of
"BannedUserHandler") won't be caught -- treat a low score as "worth a
manual read", not a proven verdict, and spot-check a few docs regardless of
score before concluding anything.

Usage:
    uv run scripts/python/doc_groundedness.py
    uv run scripts/python/doc_groundedness.py --project-id 'd-Rickyy-b|Python-BlackJackBot'
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from structural_ambiguity import extract_boundary_identifiers
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

# Identifiers this generic/short are almost always false-positive matches
# against ordinary English prose (e.g. a class named `Run` or a method
# named `get`) -- excluded from both the denominator and any match count so
# they can't inflate a groundedness score.
_MIN_IDENTIFIER_LEN = 5
_GENERIC_NAMES = {
    "config",
    "client",
    "server",
    "handler",
    "manager",
    "helper",
    "utils",
    "base",
    "model",
    "models",
    "test",
    "tests",
    "main",
    "index",
    "init",
}


@dataclass
class DocGroundedness:
    project_id: str
    n_real_identifiers: int
    intent_matches: int
    c4_matches: int
    intent_score: float
    c4_score: float
    n_boundary_identifiers: int
    intent_boundary_score: float
    c4_boundary_score: float
    unmatched_by_c4: list[str]
    unmatched_boundary_by_c4: list[str]


def _iter_python_source_files(repo_root: Path) -> list[Path]:
    files = []
    for path in repo_root.rglob("*.py"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)
        if any(part in _EXCLUDE_DIR_NAMES for part in rel.parts[:-1]):
            continue
        if is_test_path(rel):
            continue
        files.append(path)
    return files


def extract_real_identifiers(repo_root: Path) -> set[str]:
    identifiers: set[str] = set()
    for f in _iter_python_source_files(repo_root):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            name = None
            if (
                isinstance(node, ast.ClassDef)
                or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
            ):
                name = node.name
            if (
                name
                and len(name) >= _MIN_IDENTIFIER_LEN
                and name.lower() not in _GENERIC_NAMES
            ):
                identifiers.add(name)
    return identifiers


def _word_boundary_pattern(identifier: str) -> re.Pattern[str]:
    # Identifiers are typically CamelCase or snake_case, both of which are
    # \w-safe, so a plain \b...\b works without special-casing case.
    return re.compile(r"\b" + re.escape(identifier) + r"\b")


def score_doc(doc_text: str, identifiers: set[str]) -> tuple[int, list[str]]:
    matched = []
    for ident in identifiers:
        if _word_boundary_pattern(ident).search(doc_text):
            matched.append(ident)
    return len(matched), matched


def analyze_project(
    record: dict, repo_base: Path, docs_dir: Path | None = None
) -> DocGroundedness | None:
    project_id = record["project_id"]
    if record.get("language", "Python") != "Python":
        return None
    repo_root = repo_base / project_id
    if not repo_root.is_dir():
        print(f"  skip {project_id}: repository checkout not found at {repo_root}")
        return None

    identifiers = extract_real_identifiers(repo_root)
    if not identifiers:
        print(f"  skip {project_id}: no eligible real identifiers extracted")
        return None

    boundary_identifiers = extract_boundary_identifiers(repo_root) & identifiers

    def doc_path(kind: str) -> Path | None:
        if record.get(f"{kind}_doc_path"):
            return Path(record[f"{kind}_doc_path"])
        return docs_dir / kind / f"{project_id}.md" if docs_dir else None

    intent_path = doc_path("intent")
    c4_path = doc_path("c4")
    if not c4_path or not c4_path.is_file():
        print(f"  skip {project_id}: c4_doc_path missing/not found")
        return None

    c4_text = c4_path.read_text(encoding="utf-8", errors="replace")
    c4_n, c4_matched = score_doc(c4_text, identifiers)
    c4_boundary_n, c4_boundary_matched = (
        score_doc(c4_text, boundary_identifiers) if boundary_identifiers else (0, [])
    )

    intent_n = 0
    intent_boundary_n = 0
    if intent_path and intent_path.is_file():
        intent_text = intent_path.read_text(encoding="utf-8", errors="replace")
        intent_n, _ = score_doc(intent_text, identifiers)
        if boundary_identifiers:
            intent_boundary_n, _ = score_doc(intent_text, boundary_identifiers)

    unmatched = sorted(identifiers - set(c4_matched))
    unmatched_boundary = sorted(boundary_identifiers - set(c4_boundary_matched))
    return DocGroundedness(
        project_id=project_id,
        n_real_identifiers=len(identifiers),
        intent_matches=intent_n,
        c4_matches=c4_n,
        intent_score=intent_n / len(identifiers),
        c4_score=c4_n / len(identifiers),
        n_boundary_identifiers=len(boundary_identifiers),
        intent_boundary_score=(intent_boundary_n / len(boundary_identifiers))
        if boundary_identifiers
        else float("nan"),
        c4_boundary_score=(c4_boundary_n / len(boundary_identifiers))
        if boundary_identifiers
        else float("nan"),
        unmatched_by_c4=unmatched,
        unmatched_boundary_by_c4=unmatched_boundary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark-input", type=Path, default=DEFAULT_BENCHMARK_INPUT)
    parser.add_argument("--repository-base", type=Path, default=DEFAULT_REPOSITORY_BASE)
    parser.add_argument(
        "--docs-dir",
        type=Path,
        help="dir with intent/ and c4/ subdirs of <project_id>.md, for records "
        "without *_doc_path (e.g. data/out/llm_results/<model>/ctx95)",
    )
    parser.add_argument("--project-id", help="Only analyze this project_id")
    parser.add_argument(
        "--show-unmatched",
        action="store_true",
        help="Print real identifiers the c4_doc never mentions",
    )
    args = parser.parse_args()

    records = []
    with args.benchmark_input.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if args.project_id and record["project_id"] != args.project_id:
                continue
            records.append(record)

    results = []
    for record in records:
        result = analyze_project(record, args.repository_base, args.docs_dir)
        if result:
            results.append(result)

    results.sort(key=lambda r: r.c4_boundary_score)

    header = (
        f"{'project_id':<32} {'n_ids':>6} {'intent':>7} {'c4':>7} "
        f"{'n_bound':>8} {'bound_intent':>13} {'bound_c4':>9}"
    )
    print(f"\n{header}  (bound_* = restricted to real cross-component interface names)")
    print("-" * (len(header) + 45))
    for r in results:
        bound_intent = (
            f"{r.intent_boundary_score:.2f}" if r.n_boundary_identifiers else "n/a"
        )
        bound_c4 = f"{r.c4_boundary_score:.2f}" if r.n_boundary_identifiers else "n/a"
        print(
            f"{r.project_id:<32} {r.n_real_identifiers:>6} {r.intent_score:>7.2f} {r.c4_score:>7.2f} "
            f"{r.n_boundary_identifiers:>8} {bound_intent:>13} {bound_c4:>9}"
        )
        if args.show_unmatched and r.unmatched_boundary_by_c4:
            preview = ", ".join(r.unmatched_boundary_by_c4[:15])
            more = (
                f" (+{len(r.unmatched_boundary_by_c4) - 15} more)"
                if len(r.unmatched_boundary_by_c4) > 15
                else ""
            )
            print(f"  boundary names never mentioned by c4_doc: {preview}{more}")
        if args.show_unmatched and r.unmatched_by_c4:
            preview = ", ".join(r.unmatched_by_c4[:15])
            more = (
                f" (+{len(r.unmatched_by_c4) - 15} more)"
                if len(r.unmatched_by_c4) > 15
                else ""
            )
            print(f"  never mentioned by c4_doc: {preview}{more}")


if __name__ == "__main__":
    main()
