"""Run collect-kit over the Commit0 libraries and build a corpus-compatible dataset.

Two stages (run both with `all`, or separately):

  collect  For every instance in commit0_instances.jsonl (lite first), run
           collect-kit/collect.sh (Arcan in docker + collect_repo.py) on the
           prefetched checkout data/repository/commit-0|<name> (which must be
           at reference_commit -- see scripts/bash/prefetch_commit0.sh).
           Sequential by default, resumable (repos whose per-repo JSONL exists
           are skipped), keeps going on failures, and appends one status row
           per attempt to <collect_dir>/status.jsonl. Arcan raw output is kept
           under <collect_dir>/arcan/arcanOutput/<project_id>.

  combine  Merge the per-repo records into one dataset JSONL with the same
           schema as the main corpus (reverse-se_v0.6_p10-95_ctx95_dataset):
           graphs are rebuilt from the kept Arcan output with
           combine_arcan_w_context.process_repo (identical normalization and
           edge_compression_ratio), `info` is filled from the instances file
           (+ upstream GitHub metadata), NO node-count percentile filter is
           applied (the 54 are a fixed set), and oversized context files are
           dropped with the ctx95 per-category token thresholds of the MAIN
           corpus. Those thresholds are recomputed exactly as
           combine_arcan_w_context.py did at build time (p95 over the context
           files of every repo it processed, i.e. the rows of
           reverse-se_v0.6_p10-95_ctx95_stats.csv, tokens from
           data/interim/repo_context_files_all.jsonl) and verified against the
           released main-corpus dataset (no kept file exceeds them). They are
           cached in <collect_dir>/ctx95_thresholds.json.

Usage:
    uv run scripts/python/commit0_collect.py all
    uv run scripts/python/commit0_collect.py collect --xmx 5g --timeout 5400
    uv run scripts/python/commit0_collect.py collect --only babel cachetools
    uv run scripts/python/commit0_collect.py combine
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combine_arcan_w_context import (  # noqa: E402
    compute_percentile,
    count_tokens,
    filter_context_files,
    process_repo,
)

_DEFAULT_INSTANCES = ROOT / "data/raw/commit0/commit0_instances.jsonl"
_DEFAULT_REPO_BASE = ROOT / "data/repository"
_DEFAULT_COLLECT_DIR = ROOT / "data/raw/commit0/collect"
_DEFAULT_OUTPUT = ROOT / "data/out/commit0_v0.6_ctx95_dataset.jsonl"
_DEFAULT_MAIN_DATASET = ROOT / "data/out/reverse-se_v0.6_p10-95_ctx95_dataset.jsonl"
_DEFAULT_MAIN_STATS = ROOT / "data/out/reverse-se_v0.6_p10-95_ctx95_stats.csv"
_DEFAULT_MAIN_CONTEXT = ROOT / "data/interim/repo_context_files_all.jsonl"
_COLLECT_SH = ROOT / "collect-kit/collect.sh"
_CTX_P_HIGH = 95.0
_LANGUAGE_LABEL = "Python"  # main corpus stores GitHub's language name in info.language


# --- Helpers ---


def load_instances(path: Path, only: list[str] | None = None) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if only:
        wanted = set(only)
        rows = [r for r in rows if r["name"] in wanted or r["project_id"] in wanted]
    return rows


def _paths(collect_dir: Path, name: str) -> tuple[Path, Path]:
    return collect_dir / "jsonl" / f"{name}.jsonl", collect_dir / "logs" / f"{name}.log"


def _append_status(collect_dir: Path, row: dict[str, Any]) -> None:
    with open(collect_dir / "status.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _failure_reason(log_text: str, returncode: int | None) -> str:
    patterns = [
        (r"java.lang.OutOfMemoryError[^\n]*", None),
        (r"GC overhead limit exceeded", None),
        (r"Error: Arcan did not produce output[^\n]*", None),
        (r"No \.graphml file found[^\n]*", None),
        (r"Traceback \(most recent call last\)", "collect_repo.py traceback"),
        (r"Exception in thread[^\n]*", None),
    ]
    for pat, label in patterns:
        m = re.search(pat, log_text)
        if m:
            return label or m.group(0).strip()[:300]
    if returncode in (137, -9):
        return f"killed (exit {returncode}, likely docker OOM)"
    tail = [ln for ln in log_text.strip().splitlines() if ln.strip()][-1:]
    return f"exit {returncode}: {tail[0][:300] if tail else ''}"


# --- Stage 1: collect ---


def collect_one(
    inst: dict[str, Any],
    repo_base: Path,
    collect_dir: Path,
    xmx: str,
    docker_memory: str | None,
    timeout: int,
) -> dict[str, Any]:
    name, pid = inst["name"], inst["project_id"]
    out_path, log_path = _paths(collect_dir, name)
    repo_path = repo_base / pid
    arcan_dir = collect_dir / "arcan"
    row: dict[str, Any] = {"project_id": pid, "name": name, "lite": inst.get("lite")}

    if not (repo_path / ".git").is_dir():
        return {
            **row,
            "status": "fail",
            "duration_s": 0,
            "error": "repo not prefetched",
        }

    # Clear stale (possibly partial) Arcan output so collect_repo never reads it.
    shutil.rmtree(arcan_dir / "arcanOutput" / pid, ignore_errors=True)
    tmp_out = out_path.with_suffix(".jsonl.tmp")
    container = f"commit0-arcan-{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}"
    docker_args = f"--name {container}"
    if docker_memory:
        docker_args += f" --memory {docker_memory}"
    env = {
        **os.environ,
        "JAVA_TOOL_OPTIONS": f"-Xmx{xmx} -XX:+ExitOnOutOfMemoryError",
        "ARCAN_DOCKER_ARGS": docker_args,
    }
    cmd = [
        "bash",
        str(_COLLECT_SH),
        "--repo", str(repo_path),
        "--language", "PYTHON",
        "--output", str(tmp_out),
        "--arcan-output", str(arcan_dir),
        "--keep-arcan-output",
    ]  # fmt: skip

    start = time.monotonic()
    returncode: int | None = None
    timed_out = False
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "kill", container], capture_output=True)
            proc.kill()
            proc.wait()
    duration = round(time.monotonic() - start, 1)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)

    if (
        not timed_out
        and returncode == 0
        and tmp_out.exists()
        and tmp_out.stat().st_size
    ):
        tmp_out.replace(out_path)
        return {**row, "status": "ok", "duration_s": duration, "error": None}

    tmp_out.unlink(missing_ok=True)
    if timed_out:
        error = f"timeout after {timeout}s"
    else:
        error = _failure_reason(log_path.read_text(errors="replace"), returncode)
    return {**row, "status": "fail", "duration_s": duration, "error": error}


def run_collect(args: argparse.Namespace) -> None:
    collect_dir: Path = args.collect_dir
    for sub in ("jsonl", "logs", "arcan"):
        (collect_dir / sub).mkdir(parents=True, exist_ok=True)

    instances = load_instances(args.instances, args.only)
    todo = []
    for inst in instances:
        out_path, _ = _paths(collect_dir, inst["name"])
        if out_path.exists() and out_path.stat().st_size and not args.force:
            print(f"[SKIP] {inst['project_id']} (output exists)")
        else:
            todo.append(inst)
    print(f"{len(todo)} repos to collect ({len(instances) - len(todo)} skipped)")

    def _run(inst):
        return collect_one(
            inst,
            args.repo_base,
            collect_dir,
            args.xmx,
            args.docker_memory,
            args.timeout,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run, inst): inst for inst in todo}
        for fut in as_completed(futures):
            row = fut.result()
            row["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            _append_status(collect_dir, row)
            print(
                f"[{row['status'].upper():4}] {row['project_id']} "
                f"{row['duration_s']}s {row['error'] or ''}",
                flush=True,
            )


# --- Stage 2: combine ---


def _github_tokens() -> list[str]:
    """Candidate tokens: $GITHUB_TOKEN, .env's GITHUB_TOKEN, then `gh auth token`."""
    tokens = [os.environ.get("GITHUB_TOKEN", "")]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?GITHUB_TOKEN\s*=\s*(.*)", line)
            if m:
                tokens.append(m.group(1).strip().strip("'\""))
    try:
        gh = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        tokens.append(gh.stdout.strip())
    except OSError:
        pass
    return list(dict.fromkeys(t for t in tokens if t))


def _github_get(full: str, tokens: list[str]) -> dict[str, Any]:
    """GET /repos/{full}, dropping tokens that are rejected (401)."""
    import urllib.error

    while True:
        token = tokens[0] if tokens else None
        req = urllib.request.Request(
            f"https://api.github.com/repos/{full}",
            headers={"Accept": "application/vnd.github+json"}
            | ({"Authorization": f"Bearer {token}"} if token else {}),
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 401 and token:
                tokens.pop(0)
                continue
            raise


def fetch_upstream_info(
    instances: list[dict[str, Any]], cache_path: Path
) -> dict[str, dict[str, Any]]:
    """GitHub metadata of each *original* upstream repo, cached as JSONL."""
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = {r["original_repo"]: r for r in map(json.loads, f)}
    missing = [i["original_repo"] for i in instances if i["original_repo"] not in cache]
    tokens = _github_tokens() if missing else []
    for full in missing:
        try:
            data = _github_get(full, tokens)
        except Exception as e:  # leave fields null on any API failure
            print(f"  GitHub API failed for {full}: {e}")
            continue
        cache[full] = {
            "original_repo": full,
            "description": data.get("description"),
            "archived": data.get("archived"),
            "pushed_at": data.get("pushed_at"),
            "stargazers_count": data.get("stargazers_count"),
        }
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(cache[full]) + "\n")
    return cache


def compute_main_thresholds(
    stats_csv: Path, context_jsonl: Path, ctx_p_high: float
) -> dict[str, float]:
    """Replicate combine_arcan_w_context's ctx thresholds for the main corpus build."""
    import csv

    with open(stats_csv, encoding="utf-8") as f:
        processed = {r["repo"] for r in csv.DictReader(f)}
    tokens: dict[str, list[int]] = {}
    seen = 0
    with open(context_jsonl, encoding="utf-8") as f:
        for line in f:
            rc = json.loads(line)
            if rc["repo"].strip() not in processed:
                continue
            seen += 1
            for doc in rc.get("documents", []):
                t = doc.get("tokens") or count_tokens(doc.get("content"))
                tokens.setdefault(doc.get("category", ""), []).append(t)
    print(
        f"  {seen}/{len(processed)} processed main-corpus repos found in {context_jsonl.name}"
    )
    return {cat: compute_percentile(v, ctx_p_high) for cat, v in tokens.items()}


def verify_thresholds(main_dataset: Path, thresholds: dict[str, float]) -> None:
    """Check no main-corpus context file exceeds the thresholds; report tightness."""
    max_tok: dict[str, int] = {}
    count: dict[str, int] = {}
    over = 0
    with open(main_dataset, encoding="utf-8") as f:
        for line in f:
            for cf in json.loads(line)["context_files"]:
                cat, t = cf.get("category") or "", cf.get("tokens", 0)
                max_tok[cat] = max(max_tok.get(cat, 0), t)
                count[cat] = count.get(cat, 0) + 1
                if cat in thresholds and t > thresholds[cat]:
                    over += 1
    print(f"  verification vs {main_dataset.name}:")
    for cat in sorted(max_tok):
        print(
            f"    {cat:12} files={count[cat]:6}  max_kept={max_tok[cat]:7}"
            f"  threshold={thresholds.get(cat, float('nan')):9.1f}"
        )
    if over:
        raise SystemExit(f"  {over} main-corpus context files exceed the thresholds!")
    print("  OK: no main-corpus context file exceeds its category threshold")


def load_thresholds(args: argparse.Namespace) -> dict[str, float]:
    cache = args.collect_dir / "ctx95_thresholds.json"
    if cache.exists() and not args.recompute_thresholds:
        thresholds = json.loads(cache.read_text())["thresholds"]
        print(f"Loaded ctx thresholds from {cache}")
        return thresholds
    print("Computing main-corpus ctx thresholds...")
    thresholds = compute_main_thresholds(
        args.main_stats, args.main_context, _CTX_P_HIGH
    )
    verify_thresholds(args.main_dataset, thresholds)
    cache.write_text(
        json.dumps(
            {
                "ctx_p_high": _CTX_P_HIGH,
                "source": [str(args.main_stats), str(args.main_context)],
                "verified_against": str(args.main_dataset),
                "thresholds": thresholds,
            },
            indent=2,
        )
    )
    return thresholds


def run_combine(args: argparse.Namespace) -> None:
    collect_dir: Path = args.collect_dir
    instances = load_instances(args.instances)
    thresholds = load_thresholds(args)
    print(
        f"Context file token thresholds (main corpus p{_CTX_P_HIGH:.0f} per category):"
    )
    for cat, thr in sorted(thresholds.items()):
        print(f"  {cat}: {thr:.0f}")

    upstream = fetch_upstream_info(instances, collect_dir / "upstream_info.jsonl")
    arcanout_dir = collect_dir / "arcan" / "arcanOutput"

    kept, missing, total_dropped = 0, [], 0
    dropped_by_cat: dict[str, int] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as outf:
        for inst in instances:
            pid = inst["project_id"]
            out_path, _ = _paths(collect_dir, inst["name"])
            if not out_path.exists():
                missing.append(pid)
                continue
            collected = json.loads(out_path.read_text(encoding="utf-8"))
            up = upstream.get(inst["original_repo"], {})
            info_map = {
                pid: {
                    "url": inst["url"],
                    "description": up.get("description"),
                    "language": _LANGUAGE_LABEL,
                    "archived": up.get("archived"),
                    "pushed_at": up.get("pushed_at"),
                    "stargazers_count": up.get("stargazers_count"),
                }
            }
            outcome = process_repo(
                {"repo": pid, "documents": collected["context_files"]},
                arcanout_dir,
                info_map,
            )
            if outcome is None:
                missing.append(f"{pid} (no Arcan graph)")
                continue
            result, _ = outcome
            result["info"].update(
                original_repo=inst["original_repo"],
                reference_commit=inst["reference_commit"],
                lite=inst["lite"],
            )
            before = {cf["path"]: cf["category"] for cf in result["context_files"]}
            result["context_files"], dropped = filter_context_files(
                result["context_files"], thresholds
            )
            if dropped:
                after = {cf["path"] for cf in result["context_files"]}
                for path, cat in before.items():
                    if path not in after:
                        dropped_by_cat[cat] = dropped_by_cat.get(cat, 0) + 1
                        print(f"  dropped {pid}: {path} ({cat})")
            total_dropped += dropped
            json.dump(result, outf, ensure_ascii=False)
            outf.write("\n")
            kept += 1

    print(f"Wrote {kept} rows to {args.output}")
    print(f"Dropped {total_dropped} context files over threshold: {dropped_by_cat}")
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(missing)}")


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run collect-kit over Commit0 libraries and combine into a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stage", choices=["collect", "combine", "all"])
    parser.add_argument("--instances", type=Path, default=_DEFAULT_INSTANCES)
    parser.add_argument("--repo_base", type=Path, default=_DEFAULT_REPO_BASE)
    parser.add_argument("--collect_dir", type=Path, default=_DEFAULT_COLLECT_DIR)
    # collect
    parser.add_argument("--only", nargs="*", help="Restrict to these names/project_ids")
    parser.add_argument(
        "--force", action="store_true", help="Re-collect existing outputs"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--xmx", default="5g", help="Arcan JVM max heap")
    parser.add_argument("--docker_memory", default=None, help="docker run --memory")
    parser.add_argument("--timeout", type=int, default=5400, help="Per-repo seconds")
    # combine
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--main_dataset", type=Path, default=_DEFAULT_MAIN_DATASET)
    parser.add_argument("--main_stats", type=Path, default=_DEFAULT_MAIN_STATS)
    parser.add_argument("--main_context", type=Path, default=_DEFAULT_MAIN_CONTEXT)
    parser.add_argument("--recompute_thresholds", action="store_true")
    args = parser.parse_args()

    if args.stage in ("collect", "all"):
        run_collect(args)
    if args.stage in ("combine", "all"):
        run_combine(args)


if __name__ == "__main__":
    main()
