#!/usr/bin/env python3
"""
Cross-artefact alignment judge: traces intent -> requirements and
requirements -> architecture for matched, cross-generator, shuffled and
planted pairs (see reverse_se/evaluation/judge/alignment.py).

Usage (pilot: all conditions on 30 repos, validates the measure):
    uv run python -m reverse_se.evaluation.alignment_judge run \\
        --dataset data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_100_stratified.jsonl \\
        --n_repos 30 \\
        --output data/out/judge/alignment/pilot30_align-v2_gpt-6-luna.jsonl

Full run (matched pairs only, Batch API):
    uv run python -m reverse_se.evaluation.alignment_judge run \\
        --dataset data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000_stratified.jsonl \\
        --conditions matched --batch \\
        --output data/out/judge/alignment/1000stratified_align-v2_gpt-6-luna.jsonl

    uv run python -m reverse_se.evaluation.alignment_judge summary \\
        --scores data/out/judge/alignment/pilot30_align-v2_gpt-6-luna.jsonl
"""

import argparse
import ast
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
import pandas as pd
from dotenv import load_dotenv
from litellm.utils import type_to_response_format_param
from scipy.stats import wilcoxon
from tqdm import tqdm

from reverse_se.evaluation.judge.alignment import (
    ALIGN_VERSION,
    CHAIN_CONDITIONS,
    CONDITIONS,
    SCHEMAS,
    ChainJob,
    PairJob,
    alignment_metrics,
    build_chain_jobs,
    build_jobs,
    chain_rows,
    job_documents,
    render_chain_messages,
    render_messages,
)
from reverse_se.evaluation.judge.batch import (
    fetch_batch_results,
    poll_batch,
    resolve_batch_model,
    submit_batches,
)

load_dotenv()

MODELS = [
    "hosted_vllm--gemma-4-31b",
    "hosted_vllm--qwen-3.5-122b",
    "hosted_vllm--gpt-oss-120b",
]
METRICS = ["item_coverage", "req_traced", "fr_traced", "nfr_traced"]


def load_repos(
    dataset: Path, n: int | None, seed: int
) -> tuple[list[str], dict[str, str]]:
    """Repository IDs of the dataset (a seeded sample of n, if given) and
    their languages."""
    languages = {}
    for line in dataset.open():
        r = json.loads(line)
        info = r["info"] if isinstance(r["info"], dict) else ast.literal_eval(r["info"])
        languages[r["repo_id"]] = info.get("language") or ""
    repos = sorted(languages)
    if n is not None:
        repos = sorted(random.Random(seed).sample(repos, n))
    return repos, languages


def sampling(args) -> dict:
    """Unset parameters are not sent, so the model's defaults apply."""
    return {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}


def is_chain(args) -> bool:
    return "chain" in SCHEMAS[args.align_version]


def build_request(job: PairJob | ChainJob, args) -> tuple[list[dict], type, str]:
    """(messages, response schema, requirements text) for a job."""
    if isinstance(job, ChainJob):
        intent, req, c4 = job.documents(args.results_dir)
        messages = render_chain_messages(intent, req, c4, args.align_version)
        return messages, SCHEMAS[args.align_version]["chain"], req
    up, down, req_doc = job_documents(args.results_dir, job)
    messages = render_messages(job.pair, up, down, args.align_version)
    return messages, SCHEMAS[args.align_version][job.pair], req_doc


def result_rows(
    job: PairJob | ChainJob, raw: str, usage: dict, req_doc: str, args
) -> list[dict]:
    meta = {
        "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort,
        "align_version": args.align_version,
        "run": args.run,
    }
    if isinstance(job, ChainJob):
        evaluation = SCHEMAS[args.align_version]["chain"].model_validate_json(raw)
        rows = [{**row, **meta} for row in chain_rows(job, evaluation, req_doc)]
        # One call, two rows: the usage and the raw answer go on the first
        # row only, so token sums count each call once.
        rows[0] |= {"evaluation": evaluation.model_dump(), "_usage": usage}
        rows[1] |= {"_usage": {}}
        return rows
    evaluation = SCHEMAS[args.align_version][job.pair].model_validate_json(raw)
    return [
        {
            **job.record(),
            **meta,
            **alignment_metrics(job.pair, evaluation, req_doc, job.planted_ids),
            "evaluation": evaluation.model_dump(),
            "_usage": usage,
        }
    ]


def judge_live(job: PairJob | ChainJob, args) -> list[dict]:
    messages, schema, req_doc = build_request(job, args)
    response = litellm.completion(
        model=args.judge_model,
        messages=messages,
        response_format=schema,
        timeout=args.request_timeout,
        num_retries=3,
        # LiteLLM's model map does not list reasoning_effort for every
        # OpenAI model; let it through rather than drop or reject it.
        allowed_openai_params=list(sampling(args)),
        **sampling(args),
    )
    usage = json.loads(json.dumps(dict(response.usage), default=vars))
    return result_rows(job, response.choices[0].message.content, usage, req_doc, args)


def run_live(pending: list[PairJob], args, out) -> int:
    lock = threading.Lock()
    failures = 0
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(judge_live, j, args): j for j in pending}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] {futures[fut].key}: {e}")
                continue
            with lock:
                out.writelines(json.dumps(row) + "\n" for row in rows)
                out.flush()
    return failures


def run_batch(pending: list[PairJob], args, out) -> int:
    """Same requests as run_live, through the OpenAI Batch API (~50% cheaper).
    custom_id is the index in `pending`, which is rebuilt identically on a
    resume with --batch_id."""
    model = resolve_batch_model(args.judge_model)
    batch_dir = args.output.parent / (args.output.stem + "_batch")
    requests = [build_request(j, args) for j in pending]
    if args.batch_id:
        batch_ids = [b.strip() for b in args.batch_id.split(",") if b.strip()]
    else:
        lines = [
            {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    **sampling(args),
                    "response_format": type_to_response_format_param(schema),
                },
            }
            for i, (messages, schema, _) in enumerate(requests)
        ]
        batch_ids = submit_batches(args.judge_model, [], batch_dir, lines=lines)
        print(f"Submitted {len(batch_ids)} batch(es) -> {batch_dir}")
        if args.batch_submit_only:
            print(
                "Collect later by rerunning with --batch_id "
                f"$(paste -sd, {batch_dir / 'batch_ids.txt'})"
            )
            return 0

    def on_update(batch) -> None:
        c = batch.request_counts
        print(
            f"  batch {batch.id}: {batch.status} "
            f"{c.completed}/{c.total} done, {c.failed} failed",
            flush=True,
        )

    failures = 0
    for batch_id in batch_ids:
        batch = poll_batch(batch_id, poll_interval=60, on_update=on_update)
        for idx, (raw, usage, error) in fetch_batch_results(batch).items():
            job, req_doc = pending[idx], requests[idx][2]
            try:
                if error:
                    raise RuntimeError(error)
                rows = result_rows(job, raw, usage, req_doc, args)
                out.writelines(json.dumps(row) + "\n" for row in rows)
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] {job.key}: {e}")
        out.flush()
    return failures


def run(args) -> None:
    repos, languages = load_repos(args.dataset, args.n_repos, args.seed)
    build = build_chain_jobs if is_chain(args) else build_jobs
    default = CHAIN_CONDITIONS if is_chain(args) else CONDITIONS
    jobs = build(
        args.results_dir,
        args.models,
        repos,
        languages,
        seed=args.seed,
        conditions=tuple(args.conditions or default),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.open():
            r = json.loads(line)
            if r.get("run") != args.run:
                continue
            if "chain_key" in r:
                done.add(tuple(r["chain_key"]))
            else:
                done.add(
                    (
                        r["pair"],
                        r["condition"],
                        r["up_model"],
                        r["up_repo"],
                        r["down_model"],
                        r["down_repo"],
                    )
                )
    pending = [j for j in jobs if j.key not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(repos)} repos, {len(jobs)} jobs, {len(pending)} pending")
    with args.output.open("a") as out:
        failures = (run_batch if args.batch else run_live)(pending, args, out)
    print(f"done, {failures} failures")


def summary(args) -> None:
    df = pd.DataFrame([json.loads(line) for f in args.scores for line in f.open()])
    df["prompt_tokens"] = df["_usage"].map(lambda u: u.get("prompt_tokens", 0))
    df["completion_tokens"] = df["_usage"].map(lambda u: u.get("completion_tokens", 0))
    cols = METRICS + ["omitted_share", "n_items", "mean_links"]

    print("== Mean per pair and condition")
    print(df.groupby(["pair", "condition"])[cols].mean().round(3).to_string())

    print("\n== Mean per pair, condition and upstream generator")
    print(
        df.groupby(["pair", "condition", "up_model"])[METRICS]
        .mean()
        .round(3)
        .to_string()
    )

    if df.run.nunique() > 1:
        print("\n== Run-to-run spread (matched): SD across runs, averaged over pairs")
        m = df[df.condition == "matched"]
        sd = (
            m.groupby(["pair", "up_model", "up_repo"])[METRICS]
            .std()
            .groupby("pair")
            .mean()
        )
        print(sd.round(3).to_string())

    # Matched vs baselines, paired on (generator, repository).
    print("\n== Paired comparisons with matched (Wilcoxon signed-rank)")
    for pair in sorted(df.pair.unique()):
        d = df[df.pair == pair]
        matched = (
            d[d.condition == "matched"].groupby(["up_model", "up_repo"])[METRICS].mean()
        )
        for cond in ("shuffled", "cross"):
            other = (
                d[d.condition == cond].groupby(["up_model", "up_repo"])[METRICS].mean()
            )
            for m in METRICS:
                joined = pd.concat(
                    [matched[m].rename("a"), other[m].rename("b")], axis=1
                ).dropna()
                if len(joined) < 5 or (joined.a == joined.b).all():
                    continue
                p = wilcoxon(joined.a, joined.b).pvalue
                print(
                    f"{pair:11s} matched vs {cond:8s} {m:14s} "
                    f"{joined.a.mean():.3f} vs {joined.b.mean():.3f}  "
                    f"n={len(joined)}  p={p:.2g}"
                )

    # Pair-wise rubrics plant in a separate condition, the chain rubric in
    # its matched calls.
    planted = df[df["planted_ids"].map(bool)]
    if len(planted):
        print("\n== Planted requirements")
        print(
            planted.groupby("pair")[["planted_detected", "req_traced"]]
            .mean()
            .round(3)
            .to_string()
        )
        print(
            "(planted_detected: share of transplanted requirements left "
            "untraced; req_traced: original requirements in the same docs)"
        )

    print("\n== Tokens per call")
    print(
        df.groupby("pair")[["prompt_tokens", "completion_tokens"]]
        .mean()
        .round(0)
        .to_string()
    )
    pin, pout = args.price_in, args.price_out
    cost = (df.prompt_tokens.sum() * pin + df.completion_tokens.sum() * pout) / 1e6
    if args.latex:
        write_latex_table(df, args.latex)
        print(f"wrote {args.latex}")
    if args.macros:
        write_macros(df, args.macros)
        print(f"wrote {args.macros}")
    calls = int((df.prompt_tokens > 0).sum())  # chain calls carry usage on one row
    print(f"{calls} calls, ${cost:.2f} at ${pin}/M in, ${pout}/M out (live)")


# The manuscript's generator label macros
GENERATOR_LABELS = {
    "hosted_vllm--gemma-4-31b": "\\mGemma",
    "hosted_vllm--qwen-3.5-122b": "\\mQwen",
    "hosted_vllm--gpt-oss-120b": "\\mGpto",
}


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    """Table body for the paper: coverage (%) of matched pairs per
    generator, then the cross and shuffled baselines pooled over generators."""
    # Coverage only: requirement-side tracing (fr/nfr_traced) sits near the
    # ceiling with ~6 FRs per document, so it says little per generator.
    cols = [("intent_req", "item_coverage"), ("req_c4", "item_coverage")]

    def cells(d: pd.DataFrame) -> str:
        return " & ".join(f"{100 * d[d.pair == p][m].mean():.0f}" for p, m in cols)

    lines = []
    for model, label in GENERATOR_LABELS.items():
        d = df[(df.condition == "matched") & (df.up_model == model)]
        if len(d):
            lines.append(f"{label} & {cells(d)} \\\\")
    lines.append("\\midrule")
    lines.append(f"Cross-gen. & {cells(df[df.condition == 'cross'])} \\\\")
    lines.append(f"Shuffled & {cells(df[df.condition == 'shuffled'])} \\\\")
    path.write_text("\n".join(lines) + "\n")


def write_macros(df: pd.DataFrame, path: Path) -> None:
    """Macros for the paper text: coverage (%) of matched pairs as the range
    over generators, and of the shuffled baseline pooled over generators."""
    macros = [
        "% Generated by reverse_se.evaluation.alignment_judge summary -- do not "
        "edit by hand."
    ]
    for pair, name in (("intent_req", "Intent"), ("req_c4", "Arch")):
        d = df[df.pair == pair]
        per_gen = (
            100 * d[d.condition == "matched"].groupby("up_model").item_coverage.mean()
        )
        shuffled = 100 * d[d.condition == "shuffled"].item_coverage.mean()
        macros += [
            f"\\newcommand{{\\Align{name}CovMin}}{{{per_gen.min():.0f}\\xspace}}",
            f"\\newcommand{{\\Align{name}CovMax}}{{{per_gen.max():.0f}\\xspace}}",
            f"\\newcommand{{\\Align{name}CovShuffled}}{{{shuffled:.0f}\\xspace}}",
        ]
    path.write_text("\n".join(macros) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", type=Path, required=True)
    r.add_argument("--results_dir", type=Path, default=Path("data/out/llm_results"))
    r.add_argument("--output", type=Path, required=True)
    r.add_argument(
        "--n_repos",
        type=int,
        default=None,
        help="Seeded sample of this many repos (default: the whole dataset)",
    )
    r.add_argument("--models", nargs="+", default=MODELS)
    r.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=None,
        help=(
            f"Default: {' '.join(CHAIN_CONDITIONS)} for the chain rubric (v3, "
            f"where matched calls carry planted requirements), "
            f"{' '.join(CONDITIONS)} for the pair-wise ones"
        ),
    )
    r.add_argument("--align_version", choices=list(SCHEMAS), default=ALIGN_VERSION)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--run", type=int, default=1)
    r.add_argument("--judge_model", default="openai/gpt-6-luna")
    r.add_argument(
        "--reasoning_effort",
        default="low",
        help=(
            "none|low|medium|high, or '' to send nothing (model default). "
            "low: on a 5-repo test it halved output tokens against the "
            "default and separated matched, shuffled and planted pairs as "
            "well; none was more lenient on shuffled pairs."
        ),
    )
    r.add_argument("--request_timeout", type=float, default=600.0)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--batch", action="store_true", help="Use the OpenAI Batch API")
    r.add_argument(
        "--batch_id",
        default=None,
        help="Resume polling submitted batches (comma-separated; implies --batch)",
    )
    r.add_argument("--batch_submit_only", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--scores", type=Path, nargs="+", required=True)
    s.add_argument("--price_in", type=float, default=0.10)
    s.add_argument("--price_out", type=float, default=0.50)
    s.add_argument(
        "--latex",
        type=Path,
        default=None,
        help="Also write the paper table body (tables/alignment_rows.tex)",
    )
    s.add_argument(
        "--macros",
        type=Path,
        default=None,
        help="Also write the paper text macros (tables/alignment_numbers.tex)",
    )
    args = parser.parse_args()
    if args.cmd == "run" and (args.batch_id or args.batch_submit_only):
        args.batch = True
    run(args) if args.cmd == "run" else summary(args)


if __name__ == "__main__":
    main()
