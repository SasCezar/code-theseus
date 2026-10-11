"""Locating result files to judge, across the layouts the pipeline has used:
generation's llm_results/{model}/{context}/{task}/*.md (the answer; metadata
lives alongside as *.meta.json, reasoning as *.reasoning.txt), and the
flattened data/docs/{model}/{task}/*.md docs layout."""

import sys
from collections.abc import Iterator
from pathlib import Path

from reverse_se.evaluation.judge.rubrics import TASKS

# (model, context_budget, task, repo_id, path)
ResultTuple = tuple[str, str, str, str, Path]


def discover_results(
    results_dir: Path, task_filter: str | None, model_filter: list[str] | None
) -> Iterator[ResultTuple]:
    """Walk llm_results/{model}/{context}/{task}/*.md."""
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_filter and model_dir.name not in model_filter:
            continue
        for ctx_dir in sorted(model_dir.iterdir()):
            if not ctx_dir.is_dir():
                continue
            for task_dir in sorted(ctx_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task = task_dir.name
                if task_filter and task != task_filter:
                    continue
                if task not in TASKS:
                    continue
                for md_file in sorted(task_dir.glob("*.md")):
                    yield model_dir.name, ctx_dir.name, task, md_file.stem, md_file


def discover_docs(
    docs_dir: Path, task_filter: str | None, model_filter: list[str] | None
) -> Iterator[ResultTuple]:
    """Walk data/docs/{model}/{task}/*.md."""
    for model_dir in sorted(docs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_filter and model_dir.name not in model_filter:
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            if task_filter and task != task_filter:
                continue
            if task not in TASKS:
                continue
            for md_file in sorted(task_dir.glob("*.md")):
                yield model_dir.name, "-", task, md_file.stem, md_file


def single_file_iter(
    result_file: Path,
    results_dir: Path | None,
    task_override: str | None,
    model_override: str | None,
) -> Iterator[ResultTuple]:
    """Yield one tuple for a single result .md file, inferring model/context/
    task from its path relative to results_dir when possible."""
    model, ctx, task = "unknown", "unknown", "unknown"
    if results_dir and result_file.is_relative_to(results_dir):
        parts = result_file.relative_to(results_dir).parts
        # Expected: <model>/<context>/<task>/<repo_id>.md
        if len(parts) >= 4:
            model, ctx, task = parts[0], parts[1], parts[2]
        elif len(parts) == 3:
            model, ctx, task = parts[0], "default", parts[1]

    if model_override:
        model = model_override
    if task_override:
        task = task_override

    if task not in TASKS:
        print(
            f"[ERROR] Cannot infer task from path and no --task provided "
            f"(inferred: '{task}'). Use --task c4|req|intent.",
            file=sys.stderr,
        )
        return

    yield model, ctx, task, result_file.stem, result_file
