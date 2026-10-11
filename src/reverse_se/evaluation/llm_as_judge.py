#!/usr/bin/env python3
"""
LLM-as-Judge for software engineering artefacts.

Walks llm_results/{model}/{context}/{task}/{repo_id}.json, re-renders the
input prompt from the raw dataset, then asks a judge LLM to rate each
generated document using the rubric from the human evaluation survey
(see reverse_se/evaluation/judge/rubrics.py).

Usage:
    python -m reverse_se.evaluation.llm_as_judge \\
        --dataset data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000.jsonl \\
        --results_dir llm_results \\
        --task c4 \\
        --output evaluation/judge_scores.jsonl \\
        --judge_model openai/gpt-4.1-nano \\
        --limit 5

Rubric prompts are in prompts/judge/<version>/ (see its README), loaded by
judge/rubrics.py; every row records its rubric version. --rubric_version v2
selects the previous rubric, kept only to compare judges and prompts.
"""

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from reverse_se.evaluation.judge.batch import (
    fetch_batch_results,
    poll_batch,
    submit_batches,
)
from reverse_se.evaluation.judge.client import (
    call_judge,
    parse_judge_response,
)
from reverse_se.evaluation.judge.discovery import (
    ResultTuple,
    discover_docs,
    discover_results,
    single_file_iter,
)
from reverse_se.evaluation.judge.report import print_summary, write_csv
from reverse_se.evaluation.judge.rubrics import (
    RUBRIC_VERSION,
    RUBRIC_VERSIONS,
    TASKS,
    resolve_rubric,
)
from reverse_se.generation.utils import build_full_prompt, load_dataset

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-Judge for SE artefacts")
    parser.add_argument(
        "--dataset", required=True, type=Path, help="Path to the raw JSONL dataset"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Root of llm_results layout: {model}/{context}/{task}/*.md",
    )
    parser.add_argument(
        "--docs_dir",
        type=Path,
        default=None,
        help="Root of canonical docs layout: data/docs/{model}/{task}/*.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("judge_scores.jsonl"),
        help="Output JSONL file for judge scores",
    )
    parser.add_argument(
        "--judge_model",
        default="openai/gpt-4.1",
        help="LiteLLM model string for the judge (default: openai/gpt-4.1)",
    )
    parser.add_argument(
        "--api_base",
        default=None,
        help="Judge endpoint for self-hosted models, e.g. http://localhost:8004/v1",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=600.0,
        help="Seconds before a judge call times out (default: 600)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Judge sampling temperature (default: not sent, so the model's own "
            "default applies; OpenAI reasoning models only accept their default)"
        ),
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Judge top_p (default: unset)"
    )
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        help=(
            "Passed to the judge as reasoning_effort, e.g. 'high' for "
            "Mistral-Medium-3.5 (needs --reasoning-parser on the vLLM server)"
        ),
    )
    parser.add_argument(
        "--rubric_version",
        choices=list(RUBRIC_VERSIONS),
        default=RUBRIC_VERSION,
        help=(
            f"Rubric to judge with (default: {RUBRIC_VERSION}). 'v2' is the "
            "previous rubric, for judge/prompt comparisons only."
        ),
    )
    parser.add_argument(
        "--prompt_dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "prompts",
        help="Path to Jinja prompt templates",
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS),
        default=None,
        help="Evaluate only this task (default: all)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Evaluate only these model dir names (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N results (for quick testing)",
    )
    parser.add_argument(
        "--result_file",
        type=Path,
        default=None,
        help=(
            "Judge a single result .md file instead of walking results_dir. "
            "The model/context/task are inferred from the path "
            "(<results_dir>/<model>/<context>/<task>/<repo_id>.md) "
            "or can be overridden with --model / --task."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip results already present in the output file",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Write tidy long-format CSV for ggplot (default: same as --output with .csv)",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=None,
        help="Write the human-readable score summary here (default: same as --output with '_summary.txt' suffix)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of judge calls to run concurrently (default: 8)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Submit all pending judge calls as a single OpenAI Batch API job "
            "instead of live concurrent calls. ~50%% cheaper, turnaround up "
            "to 24h. Only supported for --judge_model openai/<model>."
        ),
    )
    parser.add_argument(
        "--batch_id",
        default=None,
        help=(
            "Resume polling already-submitted batches instead of submitting new "
            "ones: comma-separated ids, as in <output>_batch/batch_ids.txt "
            "(implies --batch)"
        ),
    )
    parser.add_argument(
        "--batch_submit_only",
        action="store_true",
        help=(
            "Submit the batches, save their ids to <output>_batch/batch_ids.txt "
            "and exit without waiting; collect the results later by rerunning "
            "the same command with --batch_id"
        ),
    )
    parser.add_argument(
        "--batch_poll_interval",
        type=int,
        default=30,
        help="Seconds between batch status polls (default: 30)",
    )
    args = parser.parse_args()
    if args.batch_id or args.batch_submit_only:
        args.batch = True

    args.sampling = {
        **({"temperature": args.temperature} if args.temperature is not None else {}),
        **({"top_p": args.top_p} if args.top_p is not None else {}),
        **(
            {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
        ),
    }
    if args.output_csv is None:
        args.output_csv = args.output.with_suffix(".csv")
    if args.summary_path is None:
        args.summary_path = args.output.with_name(args.output.stem + "_summary.txt")
    if not args.result_file and not args.results_dir and not args.docs_dir:
        parser.error("Provide one of --result_file, --results_dir, or --docs_dir")
    return args


def load_scored_keys(output_path: Path) -> set[tuple]:
    """Already-scored (model, ctx, task, repo_id, judge_model, rubric_version)
    keys, so a run can resume without re-judging."""
    done: set[tuple] = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(
                    (
                        row["model"],
                        row.get("context_budget", "-"),
                        row["task"],
                        row["repo_id"],
                        row.get("judge_model", ""),
                        row.get("rubric_version"),
                    )
                )
            except Exception:
                pass
    return done


def resolve_allowed_repos(
    all_results: list[ResultTuple], dataset: dict, limit: int | None
) -> set[str] | None:
    """When --limit is set, restrict to the first N repos common to every
    (model, ctx, task) group. Computed over the full candidate pool (not the
    unscored remainder) so a resumed run doesn't silently process far more
    than --limit repos per group over time."""
    if not limit:
        return None

    group_repos: dict[tuple, list[str]] = defaultdict(list)
    for model, ctx, task, repo_id, _path in all_results:
        if repo_id in dataset:
            group_repos[(model, ctx, task)].append(repo_id)
    if not group_repos:
        return None

    common = set(next(iter(group_repos.values())))
    for repos in group_repos.values():
        common &= set(repos)
    allowed = set(sorted(common)[:limit])
    print(
        f"  Limit {limit}: {len(allowed)} common repos across {len(group_repos)} groups",
        flush=True,
    )
    return allowed


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.dataset} ...", flush=True)
    dataset = load_dataset(args.dataset)
    print(f"  {len(dataset)} repos loaded", flush=True)

    if args.result_file:
        results_iter = single_file_iter(
            args.result_file,
            args.results_dir,
            args.task,
            args.model[0] if args.model else None,
        )
    elif args.docs_dir:
        results_iter = discover_docs(args.docs_dir, args.task, args.model)
    else:
        results_iter = discover_results(args.results_dir, args.task, args.model)

    done = load_scored_keys(args.output) if args.skip_existing else set()
    if done:
        print(f"  {len(done)} already scored, will skip", flush=True)

    all_results = list(results_iter)
    allowed_repos = resolve_allowed_repos(all_results, dataset, args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = skipped_missing = skipped_existing = skipped_empty = errors = 0

    # Filter to pending work items and render their prompts up front, so the
    # concurrent phase below only has to make judge calls.
    pending: list[
        tuple[str, str, str, str, str, str]
    ] = []  # (model, ctx, task, repo_id, input_prompt, generated_doc)
    for model, ctx, task, repo_id, md_path in all_results:
        if allowed_repos is not None and repo_id not in allowed_repos:
            continue

        key = (model, ctx, task, repo_id, args.judge_model, args.rubric_version)
        if key in done:
            skipped_existing += 1
            continue

        if repo_id not in dataset:
            skipped_missing += 1
            continue

        generated_doc = md_path.read_text(encoding="utf-8").strip()
        if not generated_doc:
            skipped_empty += 1
            continue

        try:
            input_prompt = build_full_prompt(args.prompt_dir, task, dataset[repo_id])
        except Exception as e:
            errors += 1
            print(f"  [WARN] render failed for {repo_id}/{task}: {e}", flush=True)
            continue

        pending.append((model, ctx, task, repo_id, input_prompt, generated_doc))

    if args.batch:
        count, errors = judge_via_batch(pending, args)
        finish(args, count, errors, skipped_missing, skipped_existing, skipped_empty)
        return

    write_lock = threading.Lock()
    progress = tqdm(total=len(pending), desc="Judging", unit="doc")

    def judge_one(item: tuple[str, str, str, str, str, str]) -> None:
        nonlocal count, errors
        model, ctx, task, repo_id, input_prompt, generated_doc = item
        try:
            scores = call_judge(
                args.judge_model,
                task,
                input_prompt,
                generated_doc,
                args.api_base,
                args.rubric_version,
                args.sampling,
                args.request_timeout,
            )
        except Exception as e:
            with write_lock:
                errors += 1
                tqdm.write(f"  [ERROR] {model}/{ctx}/{task}/{repo_id}: {e}")
                progress.set_postfix(
                    ok=count,
                    fail=errors,
                    skip=skipped_missing + skipped_existing + skipped_empty,
                )
                progress.update(1)
            return

        row = {
            "model": model,
            "context_budget": ctx,
            "task": task,
            "repo_id": repo_id,
            "judge_model": args.judge_model,
            "judge_sampling": args.sampling,
            "rubric_version": args.rubric_version,
            **scores,
        }
        with write_lock:
            with open(args.output, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
            count += 1
            progress.set_postfix(
                ok=count,
                fail=errors,
                skip=skipped_missing + skipped_existing + skipped_empty,
            )
            progress.update(1)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(judge_one, item) for item in pending]
        for future in as_completed(futures):
            future.result()

    progress.close()

    finish(args, count, errors, skipped_missing, skipped_existing, skipped_empty)


def finish(
    args: argparse.Namespace,
    count: int,
    errors: int,
    skipped_missing: int,
    skipped_existing: int,
    skipped_empty: int,
) -> None:
    print(
        f"\nDone. {count} scored, {errors} failed, "
        f"{skipped_missing} not in dataset, {skipped_existing} already scored, "
        f"{skipped_empty} empty docs."
    )
    print(f"Output: {args.output}")
    if args.output.exists():
        print_summary(args.output, args.summary_path)
        write_csv(args.output, args.output_csv)
        print(f"CSV: {args.output_csv}")


def judge_via_batch(
    pending: list[tuple[str, str, str, str, str, str]],
    args: argparse.Namespace,
) -> tuple[int, int]:
    """Submits (or resumes) a single OpenAI Batch API job covering every
    pending item, blocks until it completes, and appends the results to
    args.output in the same row shape the live path writes.

    Returns (count, errors)."""
    if not pending:
        print("Nothing pending to judge.", flush=True)
        return 0, 0

    batch_dir = args.output.parent / (args.output.stem + "_batch")
    if args.batch_id:
        batch_ids = [b.strip() for b in args.batch_id.split(",") if b.strip()]
        print(
            f"Resuming {len(batch_ids)} batch(es) ({len(pending)} pending items)",
            flush=True,
        )
    else:
        batch_ids = submit_batches(
            args.judge_model, pending, batch_dir, args.rubric_version, args.sampling
        )
        print(
            f"Submitted {len(batch_ids)} batch(es) ({len(pending)} requests): "
            f"{', '.join(batch_ids)} -> {batch_dir}",
            flush=True,
        )
        if args.batch_submit_only:
            print(
                f"Not waiting (--batch_submit_only). Collect the results later by "
                f"rerunning this command with --batch_id "
                f"$(paste -sd, {batch_dir / 'batch_ids.txt'})",
                flush=True,
            )
            return 0, 0

    def on_update(batch) -> None:
        counts = batch.request_counts
        tqdm.write(
            f"  batch {batch.id}: status={batch.status} "
            f"completed={counts.completed}/{counts.total} failed={counts.failed}"
        )

    results: dict = {}
    for batch_id in batch_ids:
        batch = poll_batch(
            batch_id, poll_interval=args.batch_poll_interval, on_update=on_update
        )
        if batch.status != "completed":
            print(
                f"[ERROR] batch {batch_id} ended with status={batch.status}",
                flush=True,
            )
        results.update(fetch_batch_results(batch))
    count = errors = 0
    with open(args.output, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(
            tqdm(pending, desc="Writing batch results", unit="doc")
        ):
            model, ctx, task, repo_id, _input_prompt, _generated_doc = item
            raw, usage, error = results.get(
                idx, (None, {}, "missing from batch output")
            )
            if error:
                errors += 1
                tqdm.write(f"  [ERROR] {model}/{ctx}/{task}/{repo_id}: {error}")
                continue
            _, schema = resolve_rubric(task, args.rubric_version)
            try:
                scores = parse_judge_response(raw, schema, usage)
            except Exception as e:
                errors += 1
                tqdm.write(
                    f"  [ERROR] {model}/{ctx}/{task}/{repo_id}: parse failed: {e}"
                )
                continue
            row = {
                "model": model,
                "context_budget": ctx,
                "task": task,
                "repo_id": repo_id,
                "judge_model": args.judge_model,
                "judge_sampling": args.sampling,
                "rubric_version": args.rubric_version,
                **scores,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            count += 1

    return count, errors


if __name__ == "__main__":
    main()
