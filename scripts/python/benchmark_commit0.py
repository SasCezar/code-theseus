#!/usr/bin/env python3
"""Commit0-backed A/B benchmark: intent_only vs req_arch (intent +
requirements + C4 architecture), on the 54 Commit0 libraries.

See reverse_se.benchmark.commit0 for what is reused from Commit0 and what
is added on top. Three subcommands:

  prepare  per library: pull Commit0's image, build the frozen scaffold
           (docs/docstrings/history removed, tests split visible/held-out),
           build the agent image (Commit0 image + OpenHands), and run the
           validation gates -- the reference commit must pass
           (--min-reference-pass), the scaffold must not. Writes
           <out>/scaffolds/<name>/ and <out>/scaffold_qualification.jsonl.
  run      per accepted library x condition x replicate: copy the scaffold,
           run OpenHands inside the library's agent image on the isolated
           network, then grade in a fresh container of the untouched
           Commit0 image with the complete test suite. Upserts
           <out>/results.jsonl by run_id.
  grade    re-grade existing run workspaces (e.g. after a grading fix).

Docs come from the regular generation stage:
<--docs-dir>/{intent,req,c4}/<project_id>.md, e.g.
data/out/llm_results/<model>/ctx95/intent/commit-0|tinydb.md.

Usage:
    uv run scripts/python/benchmark_commit0.py prepare --lite
    uv run scripts/python/benchmark_commit0.py run \\
        --docs-dir data/out/llm_results/hosted_vllm--qwen-3.5-122b/ctx95 \\
        --model hosted_vllm/my-coding-model --api_base http://vllm:8000/v1 \\
        --replicates 3 --max-iterations 150 --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm
from utils import ROOT_DIR

from reverse_se.benchmark import commit0
from reverse_se.benchmark.conditions import CONDITIONS, build_task
from reverse_se.benchmark.openhands_launch import run_task

DEFAULT_INSTANCES = ROOT_DIR / "data/raw/commit0/commit0_instances.jsonl"
DEFAULT_OUT = ROOT_DIR / "data/out/benchmark/commit0"
DEFAULT_NETWORK = os.environ.get("AGENT_SANDBOX_NETWORK", "reverse-se_agent-sandbox")
_HOST_ROOT = os.environ.get("HOST_PROJECT_ROOT")


def host_path(path: Path) -> Path:
    """Local path -> path the Docker daemon sees (differs inside the
    `benchmark` compose container, whose /app is HOST_PROJECT_ROOT)."""
    path = path.resolve()
    if not _HOST_ROOT:
        return path
    return Path(_HOST_ROOT) / path.relative_to(ROOT_DIR.resolve())


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _upsert_jsonl(path: Path, records: list[dict], key: str) -> int:
    existing: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record[key]] = record
    for record in records:
        existing[record[key]] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in sorted(existing.values(), key=lambda r: r[key]):
            f.write(json.dumps(record) + "\n")
    return len(existing)


def select_instances(args: argparse.Namespace) -> list[dict]:
    instances = commit0.load_instances(args.instances)
    if args.lite:
        instances = [i for i in instances if i["lite"]]
    if args.libs:
        instances = [i for i in instances if i["name"] in set(args.libs)]
    return instances


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    """Atomic (temp file + rename): sync_results, in the main thread, reads
    every summary while worker threads write theirs -- a reader must never
    see a half-written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def scaffold_dir(args, name: str) -> Path:
    return args.out / "scaffolds" / args.scaffold / name


# ── prepare ───────────────────────────────────────────────────────────────────


def prepare_one(instance: dict, args: argparse.Namespace) -> dict:
    name = instance["name"]
    dest = scaffold_dir(args, name)
    record = {
        "project_id": instance["project_id"],
        "name": name,
        "prepared_at": _utcnow(),
    }
    try:
        test_ids = commit0.fetch_test_ids(name)
        record.update(
            commit0.build_scaffold(
                instance,
                dest,
                test_ids,
                args.visible_fraction,
                args.seed,
                level=args.scaffold,
                host_path=host_path,
            )
        )
        if not args.skip_agent_image:
            record["agent_image"] = commit0.build_agent_image(
                instance, args.out / "images" / name, record["import_roots"]
            )
            record["leak_paths"] = commit0.find_leaks(
                record["agent_image"], record["import_roots"]
            )
            record["leak_checked"] = True
        if not args.skip_gates:
            held_out = _load_json(dest / "held_out.json")
            for gate, kwargs in (
                ("reference", {"reference": True}),
                ("scaffold", {"overlay": dest / "gate_overlay"}),
            ):
                if gate == "scaffold":
                    commit0.build_overlay(
                        dest / "repo",
                        record["scaffold_commit"],
                        instance,
                        kwargs["overlay"],
                        _load_json(dest / "base_impl_files.json"),
                    )
                out = dest / f"gate_{gate}"
                graded = commit0.grade(
                    instance, out, host_path, timeout=args.eval_timeout, **kwargs
                )
                scores = commit0.score(out / "report.json", test_ids, held_out)
                record[f"{gate}_gate"] = {**graded, **scores}
            ref = record["reference_gate"]["all_pass_rate"]
            stub = record["scaffold_gate"]["all_pass_rate"]
            ok = ref >= args.min_reference_pass and stub < ref
            record["status"] = "accepted" if ok else "rejected_gate"
            if not record.get("num_held_out_test_ids"):
                # Nothing to score held-out -- observed: fabric's
                # pytest-relaxed tests (no test_ prefix) weren't split.
                record["status"] = "rejected_no_held_out"
        else:
            record["status"] = "accepted_ungated"
        if record.get("leak_paths"):
            # Another copy of the library is readable in the agent's
            # container -- no way to run it without handing over the answer.
            record["status"] = "rejected_leak"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{exc}\n{traceback.format_exc()[-3000:]}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "scaffold.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def cmd_prepare(args: argparse.Namespace) -> None:
    instances = select_instances(args)
    print(f"Preparing {len(instances)} libraries -> {args.out}")
    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(prepare_one, i, args) for i in instances]
        for future in tqdm(as_completed(futures), total=len(futures), desc="prepare"):
            records.append(future.result())
    total = _upsert_jsonl(
        args.out / "scaffolds" / args.scaffold / "qualification.jsonl",
        records,
        "project_id",
    )
    for r in sorted(records, key=lambda r: r["name"]):
        ref = r.get("reference_gate", {}).get("all_pass_rate")
        stub = r.get("scaffold_gate", {}).get("all_pass_rate")
        print(
            f"  {r['name']:24} {r['status']:16} tests={r.get('num_test_ids')} "
            f"held_out={r.get('num_held_out_test_ids')} ref={ref} scaffold={stub}"
            + (f"  leaks={len(r['leak_paths'])}" if r.get("leak_paths") else "")
            + (f"  ERROR: {r['error'].splitlines()[0]}" if r.get("error") else "")
        )
    print(
        f"{total} records in {args.out / 'scaffolds' / args.scaffold / 'qualification.jsonl'}"
    )


# ── run / grade ───────────────────────────────────────────────────────────────


def _run_paths(args, name: str, condition: str, replicate: int) -> tuple[str, Path]:
    model_slug = args.model.replace("/", "__")
    run_id = f"{model_slug}_{args.scaffold}_commit0-{name}_{condition}_{replicate:02d}"
    return (
        run_id,
        args.out
        / "runs"
        / model_slug
        / args.scaffold
        / name
        / condition
        / f"{replicate:02d}",
    )


def grade_run(instance: dict, manifest: dict, run_dir: Path, args) -> dict:
    sdir = scaffold_dir(args, instance["name"])
    overlay = run_dir / "grade_overlay"
    overlay_stats = commit0.build_overlay(
        run_dir / "workspace",
        manifest["scaffold_commit"],
        instance,
        overlay,
        _load_json(sdir / "base_impl_files.json"),
    )
    graded = commit0.grade(
        instance,
        run_dir / "grade",
        host_path,
        overlay=overlay,
        timeout=args.eval_timeout,
        test_timeout=args.test_timeout,
    )
    scores = commit0.score(
        run_dir / "grade" / "report.json",
        _load_json(sdir / "test_ids.json"),
        _load_json(sdir / "held_out.json"),
    )
    tests = {**overlay_stats, **graded, **scores}
    (run_dir / "tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    return tests


def _grading_fields(tests: dict) -> dict:
    return {
        **{
            k: tests[k]
            for k in tests
            if k.endswith(("_passed", "_total", "_pass_rate"))
        },
        "tests_ran": tests["ran"],
        "grade_error": tests["error"],
        # Outcomes recovered from verbose output because pytest died before
        # writing its report (commit0.score), and the per-test limit used.
        "salvaged": tests.get("salvaged", False),
        "test_timeout": tests.get("test_timeout", 0),
        # Tests killed by the grading watchdog (counted as failed).
        "hung_tests": tests.get("hung_tests", []),
    }


def _trajectory_fields(run_dir: Path, manifest: dict, name: str) -> dict:
    trajectory = run_dir / "trajectory.jsonl"
    roots = manifest.get("import_roots", [])
    audit_flags = commit0.audit_trajectory(trajectory, roots, commit0.agent_image(name))
    searches = commit0.search_attempts(trajectory, roots)
    return {
        # False: the agent saw another installed copy of the library
        # (commit0.audit_trajectory) -- exclude the run from analysis.
        "audit_ok": not audit_flags,
        "audit_flags": audit_flags,
        # Commands hunting for the reference outside the workspace
        # (commit0.search_attempts) -- a behavior metric, not a validity flag.
        "search_attempts": len(searches),
        "search_commands": searches[:20],
        # Replies with no action (e.g. reasoning cut off by the token cap);
        # compare across conditions -- see commit0.empty_replies.
        "empty_replies": commit0.empty_replies(trajectory),
    }


def _settings(args) -> dict:
    """What makes two runs of a (library, condition, replicate) comparable;
    --skip-existing only reuses a run whose config matches these."""
    return {
        "model": args.model,
        "scaffold": args.scaffold,
        "docs_dir": str(args.docs_dir),
        "max_iterations": args.max_iterations,
        "timeout_seconds": args.timeout,
        "max_output_tokens": args.max_output_tokens,
        "no_thinking": args.no_thinking,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
    }


def _completed(run_dir: Path, args, manifest: dict) -> bool:
    """A run --skip-existing can reuse: it has a summary, the agent actually
    ran (not interrupted before or during launch -- a Ctrl-C'd run still
    writes a summary, as launch_error with 0 steps), same settings, and the
    same scaffold (a re-`prepare`d scaffold -- e.g. a changed test split --
    must not be mixed with runs on the old one)."""
    summary, config = run_dir / "summary.json", run_dir / "config.json"
    if not summary.is_file() or not config.is_file():
        return False
    result = _load_json(summary)
    # launch_error: interrupted; stuck: stopped by stuck detection before the
    # runner recovered from it (runs from before that fix) -- redo both.
    if result.get("terminal_status") in ("launch_error", "stuck") or not result.get(
        "agent_steps"
    ):
        return False
    config = _load_json(config)
    return (
        config.get("settings") == _settings(args)
        and config.get("scaffold_commit") == manifest["scaffold_commit"]
    )


def sync_results(args) -> int:
    """Rebuild <out>/results.jsonl from every run's summary.json -- the
    summaries are the source of truth, so rows can't go missing when the
    main process is interrupted between a run finishing and its row being
    written (observed after a Ctrl-C)."""
    rows = []
    for path in sorted((args.out / "runs").glob("*/*/*/*/*/summary.json")):
        try:
            rows.append(_load_json(path))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipping unreadable {path}: {exc}")
    out = args.out / "results.jsonl"
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(out)
    return len(rows)


def run_one(
    instance: dict, manifest: dict, condition: str, replicate: int, args
) -> dict:
    name = instance["name"]
    run_id, run_dir = _run_paths(args, name, condition, replicate)
    workspace = run_dir / "workspace"
    commit0.remove_tree(run_dir, host_path)
    run_dir.mkdir(parents=True)
    shutil.copytree(scaffold_dir(args, name) / "repo", workspace, symlinks=True)

    docs = {
        kind: args.docs_dir / kind / f"{instance['project_id']}.md"
        for kind in ("intent", "req", "c4")
    }
    task_text = build_task(
        commit0.task_preamble(instance, args.scaffold), docs, condition
    )
    (run_dir / "input.txt").write_text(task_text, encoding="utf-8")
    config = {
        "run_id": run_id,
        "project_id": instance["project_id"],
        "condition": condition,
        "replicate": replicate,
        "scaffold": args.scaffold,
        "docs": {kind: str(path) for kind, path in docs.items()},
        "scaffold_commit": manifest["scaffold_commit"],
        "agent_image": commit0.agent_image(name),
        "model": args.model,
        "api_base": args.api_base,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reasoning_effort": args.reasoning_effort,
        "max_iterations": args.max_iterations,
        "llm_timeout_seconds": args.llm_timeout,
        "max_output_tokens": args.max_output_tokens,
        "no_thinking": args.no_thinking,
        "timeout_seconds": args.timeout,
        "network": args.network,
        "task_tokens": commit0.task_tokens(task_text),
        "settings": _settings(args),
        "started_at": _utcnow(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    agent = run_task(
        task_text=task_text,
        workspace_dir=workspace,
        run_dir=run_dir,
        image=commit0.agent_image(name),
        network=args.network,
        host_path=host_path,
        model=args.model,
        api_key=args.api_key or "unused",
        api_base=args.api_base,
        temperature=args.temperature,
        top_p=args.top_p,
        max_iterations=args.max_iterations,
        reasoning_effort=args.reasoning_effort,
        llm_timeout=args.llm_timeout,
        command_limit=args.command_limit,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        openhands_python=commit0.AGENT_OPENHANDS_PYTHON,
        agent_shell=commit0.AGENT_SHELL,
        llm_extra_body=(
            {"chat_template_kwargs": {"enable_thinking": False}}
            if args.no_thinking
            else None
        ),
    )
    log = [f"terminal_status={agent.terminal_status}"]
    for extra in (agent.launch_error, agent.error, agent.subprocess_stderr):
        if extra:
            log.append(extra)
    (run_dir / "agent.log").write_text("\n".join(log), encoding="utf-8")

    commit0._git(["add", "-A"], workspace)
    diff = commit0._git(
        ["diff", "--cached", "--no-color", manifest["scaffold_commit"]], workspace
    )
    (run_dir / "final.patch").write_text(diff.stdout, encoding="utf-8")

    tests = grade_run(instance, manifest, run_dir, args)
    result = {
        "run_id": run_id,
        "project_id": instance["project_id"],
        "name": name,
        "lite": instance["lite"],
        "scaffold": args.scaffold,
        "condition": condition,
        "replicate": replicate,
        "agent": "openhands",
        "model": args.model,
        "docs_dir": str(args.docs_dir),
        **_grading_fields(tests),
        "reference_pass_rate": manifest.get("reference_gate", {}).get("all_pass_rate"),
        "input_tokens": agent.prompt_tokens,
        "output_tokens": agent.completion_tokens,
        "accumulated_cost": agent.accumulated_cost,
        "wall_time_seconds": agent.wall_time_seconds,
        "agent_steps": agent.agent_steps,
        "terminal_status": agent.terminal_status,
        **_trajectory_fields(run_dir, manifest, name),
        "task_tokens": config["task_tokens"],
        # Text-only stops that were nudged to continue, and whether the agent
        # ended via the finish tool (vs. step/time budget) -- see
        # openhands_runner._CONTINUE_NUDGE.
        "continuations": agent.continuations,
        "finished_by_tool": agent.finished_by_tool,
        # Stuck-detection stops recovered by a terminal-reset nudge.
        "stuck_recoveries": agent.stuck_recoveries,
        # Terminal commands killed at --command-limit.
        "commands_killed": agent.commands_killed,
        **_settings(args),
        "finished_at": _utcnow(),
    }
    _write_json(run_dir / "summary.json", result)
    return result


def _accepted(args) -> list[tuple[dict, dict]]:
    pairs = []
    for instance in select_instances(args):
        path = scaffold_dir(args, instance["name"]) / "scaffold.json"
        if not path.is_file():
            continue
        manifest = _load_json(path)
        if manifest.get("status") == "accepted" and manifest.get("leak_checked"):
            pairs.append((instance, manifest))
    return pairs


def cmd_run(args: argparse.Namespace) -> None:
    pairs = []
    for instance, manifest in _accepted(args):
        missing = [
            k
            for k in ("intent", "req", "c4")
            if not (args.docs_dir / k / f"{instance['project_id']}.md").is_file()
        ]
        if missing:
            print(
                f"  skip {instance['name']}: missing docs {missing} in {args.docs_dir}"
            )
            continue
        pairs.append((instance, manifest))
    jobs = [
        (instance, manifest, condition, replicate)
        for instance, manifest in pairs
        for condition in args.conditions
        for replicate in range(1, args.replicates + 1)
    ]
    if args.skip_existing:
        jobs = [
            j
            for j in jobs
            if not _completed(_run_paths(args, j[0]["name"], j[2], j[3])[1], args, j[1])
        ]
    print(f"Scheduling {len(jobs)} runs over {len(pairs)} libraries")
    start, results = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, *job, args) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="runs"):
            # One failure must not stop the bookkeeping: an exception here
            # ends the loop while the pool still finishes every queued run,
            # silently -- observed: progress frozen at 29/62 for 8 h.
            try:
                results.append(future.result())
            except Exception:
                print(f"\n  run failed:\n{traceback.format_exc()}")
            try:
                sync_results(args)
            except Exception:
                print(f"\n  results sync failed:\n{traceback.format_exc()}")
    print(f"Completed {len(results)} runs in {time.monotonic() - start:.0f}s")
    for r in sorted(results, key=lambda r: r["run_id"]):
        print(
            f"  {r['run_id']}: {r['terminal_status']} "
            f"held_out={r['held_out_passed']}/{r['held_out_total']} "
            f"all={r['all_passed']}/{r['all_total']} steps={r['agent_steps']}"
            + ("" if r["audit_ok"] else "  AUDIT: touched another copy of the library")
        )


def cmd_grade(args: argparse.Namespace) -> None:
    updated = []
    for instance, manifest in _accepted(args):
        for condition in args.conditions:
            for replicate in range(1, args.replicates + 1):
                run_id, run_dir = _run_paths(
                    args, instance["name"], condition, replicate
                )
                summary_path = run_dir / "summary.json"
                if not summary_path.is_file():
                    continue
                tests = grade_run(instance, manifest, run_dir, args)
                summary = _load_json(summary_path)
                summary.update(_grading_fields(tests))
                summary.update(_trajectory_fields(run_dir, manifest, instance["name"]))
                _write_json(summary_path, summary)
                updated.append(summary)
                print(
                    f"  {run_id}: held_out={summary['held_out_passed']}/{summary['held_out_total']}"
                )
    print(f"{sync_results(args)} runs in {args.out / 'results.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    common.add_argument("--out", type=Path, default=DEFAULT_OUT)
    common.add_argument(
        "--lite", action="store_true", help="Commit0 lite split only (16)"
    )
    common.add_argument("--libs", nargs="+", help="Restrict to these library names")
    common.add_argument("--jobs", type=int, default=2)
    common.add_argument(
        "--scaffold",
        choices=commit0.SCAFFOLD_LEVELS,
        default="stubs",
        help="stubs: Commit0's starter (all signatures); api: only the "
        "public interface the tests use",
    )
    common.add_argument(
        "--eval-timeout", type=int, default=1800, help="Grading timeout (s)"
    )
    common.add_argument(
        "--test-timeout",
        type=int,
        default=60,
        help="Per-test grading limit in seconds (pytest-timeout), so one "
        "hanging test fails alone instead of timing out the whole suite and "
        "zeroing the run; 0 = Commit0's eval command exactly",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", parents=[common])
    p.add_argument("--visible-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-reference-pass", type=float, default=0.9)
    p.add_argument("--skip-agent-image", action="store_true")
    p.add_argument("--skip-gates", action="store_true")
    p.set_defaults(func=cmd_prepare)

    runs = argparse.ArgumentParser(add_help=False)
    runs.add_argument("--model", required=True, help="LiteLLM model string")
    runs.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS
    )
    runs.add_argument("--replicates", type=int, default=1)

    r = sub.add_parser("run", parents=[common, runs])
    r.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="dir with intent/, req/, c4/ subdirs of <project_id>.md",
    )
    r.add_argument("--api_base", default=None)
    r.add_argument("--api_key", default=None)
    r.add_argument("--temperature", type=float, default=0.2)
    r.add_argument("--top_p", type=float, default=1.0)
    r.add_argument(
        "--reasoning_effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "none"],
    )
    r.add_argument("--max-iterations", type=int, default=1000)
    r.add_argument(
        "--timeout",
        type=int,
        default=21600,
        help="Agent wall-clock budget (s); 1000 steps took up to ~4.7 h",
    )
    r.add_argument(
        "--llm-timeout", type=int, default=1200, help="Per-request LLM timeout (s)"
    )
    r.add_argument(
        "--command-limit",
        type=int,
        default=1200,
        help="Longest a single agent terminal command may run (s) before its "
        "processes are killed -- one command that never returned froze a run "
        "for 4 h. Not a comparability setting: runs are reused across values",
    )
    r.add_argument(
        "--max-output-tokens",
        type=int,
        default=32768,
        help="Cap on tokens per model reply. Uncapped, a model starting from an "
        "empty source tree can spend 10+ minutes on one reply (whole library "
        "in one tool call, or runaway reasoning) and burn the run's time budget; "
        "too low (16k), a reasoning model's thinking alone fills it and the "
        "reply carries no action (counted as empty_replies)",
    )
    r.add_argument(
        "--no-thinking",
        action="store_true",
        help="Turn off the model's thinking (vLLM chat_template_kwargs "
        "enable_thinking=false, Qwen-style models). Applies to both conditions",
    )
    r.add_argument(
        "--network",
        default=DEFAULT_NETWORK,
        help="Docker network for the agent: must reach the model "
        "endpoint and nothing else (internal: true)",
    )
    r.add_argument("--skip-existing", action="store_true")
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("grade", parents=[common, runs])
    g.set_defaults(func=cmd_grade)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
