#!/usr/bin/env python3
"""Architectural/code quality (Arcan, pyscn) of agent-built libraries vs. the reference.

Runs Arcan on the implementation files (commit0.is_impl_file, the same set
structure_commit0.extract compares) of every valid Commit0 run and of each
library's reference checkout, then compares Intent, Full and the reference
on architectural smells, technical debt (ATDI) and coupling.

Steps (each skips work already done):
    stage    copy the implementation .py files into <work>/src/<round>/<id>
    run      Arcan (Docker) on every staged tree -> <work>/arcan/<round>/
    pyscn    pyscn on every staged tree -> <work>/pyscn/<round>/<id>.json
    analyze  per-run metrics CSV + per-library comparison (--tool arcan|pyscn)

Usage:
    uv run scripts/python/arcan_commit0.py stage
    uv run scripts/python/arcan_commit0.py run --jobs 4
    uv run scripts/python/arcan_commit0.py pyscn
    uv run scripts/python/arcan_commit0.py analyze [--tool pyscn] [--csv runs.csv]
    # rounds default to Gemma (commit0-gemma) and Q35 (commit0):
    #   --round Gemma data/out/benchmark/commit0-gemma --round Q35 ...
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_commit0 import split_valid
from scipy import stats
from structure_commit0 import reference_repo

from reverse_se.benchmark import commit0

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/arcan-tech/arcan-2-cli-trial:latest"
PYSCN = "pyscn@1.32.3"
REFERENCE = "reference"
CONDITIONS = ("intent_only", "req_arch")
ROUNDS = [
    ("Gemma", "data/out/benchmark/commit0-gemma"),
    ("Q35", "data/out/benchmark/commit0"),
]
SMELLS = {
    "cyclicDep": "cd",
    "hubLikeDep": "hl",
    "godComponent": "gc",
    "unstableDep": "ud",
}


# ── stage ─────────────────────────────────────────────────────────────────────


def copy_impl(repo: Path, instance: dict, dest: Path) -> int:
    files = [
        p.relative_to(repo).as_posix()
        for p in repo.rglob("*.py")
        if p.relative_to(repo).parts[0] not in (".venv", ".git")
        and "__pycache__" not in p.parts
        and commit0.is_impl_file(p.relative_to(repo).as_posix(), instance)
    ]
    if dest.exists():
        shutil.rmtree(dest)
    for rel in files:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dest / rel)
    return len(files)


def valid_runs(out: Path, args) -> pd.DataFrame:
    runs = pd.DataFrame(
        [
            json.loads(x)
            for x in (out / "results.jsonl").read_text().splitlines()
            if x.strip()
        ]
    )
    runs = runs[(runs["scaffold"] == args.scaffold) & ~runs["name"].isin(args.exclude)]
    valid, bad = split_valid(runs)
    print(f"{out.name}: {len(runs)} runs, {len(valid)} valid, {len(bad)} excluded")
    return valid


def run_id(name: str, condition: str, replicate: int) -> str:
    return f"{name}__{condition}__{int(replicate):02d}"


def stage(args, instances: dict) -> None:
    names = set()
    for label, out in args.round:
        for r in valid_runs(ROOT / out, args).itertuples():
            names.add(r.name)
            dest = args.work / "src" / label / run_id(r.name, r.condition, r.replicate)
            if dest.exists() and not args.force:
                continue
            ws = (
                ROOT
                / out
                / "runs"
                / r.model.replace("/", "__")
                / r.scaffold
                / r.name
                / r.condition
                / f"{int(r.replicate):02d}"
                / "workspace"
            )
            n = copy_impl(ws, instances[r.name], dest)
            if n == 0:
                print(f"  {label} {dest.name}: no implementation files")
    for name in sorted(names):
        dest = args.work / "src" / REFERENCE / name
        if not dest.exists() or args.force:
            copy_impl(reference_repo(instances[name]), instances[name], dest)
    print(f"staged under {args.work / 'src'}")


# ── run ───────────────────────────────────────────────────────────────────────


def arcan(src: Path, out_dir: Path, log: Path, args) -> tuple[str, int]:
    container = f"arcan-commit0-{out_dir.name}-{src.name}".replace("|", "_")
    cmd = ["docker", "run", "--rm", "--name", container, "--network", "none"]
    if platform.machine() == "arm64":
        cmd += ["--platform", "linux/amd64"]
    cmd += [
        "-e", f"JAVA_TOOL_OPTIONS={args.java_opts}",
        "-v", f"{src.parent}:/data/repos:ro",
        "-v", f"{out_dir}:/data/arcan_out",
        IMAGE, "analyze", "-i", f"/data/repos/{src.name}", "-o", "/data/arcan_out",
        "-l", "PYTHON", "--all",
        "output.writeDependencyGraph=true", "output.writeMembership=true",
        "output.writeDependency=true",
    ]  # fmt: skip
    # Arcan sometimes hangs on small trees (100% of one core, no output).
    with log.open("w") as fh:
        try:
            code = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout
            ).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container], capture_output=True)
            code = 124
    return src.name, code


def run(args) -> None:
    todo = []
    for label in [REFERENCE, *(lbl for lbl, _ in args.round)]:
        out_dir = args.work / "arcan" / label
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        for src in sorted((args.work / "src" / label).iterdir()):
            done = out_dir / "arcanOutput" / src.name / "component-metrics.csv"
            if not any(src.rglob("*.py")):
                continue
            if done.exists() and not args.force:
                continue
            todo.append((src, out_dir, out_dir / "logs" / f"{src.name}.log"))
    print(f"{len(todo)} analyses, {args.jobs} at a time")
    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(arcan, *t, args) for t in todo]
        for i, f in enumerate(futures, 1):
            name, code = f.result()
            print(
                f"[{i}/{len(todo)}] {todo[i - 1][1].name}/{name} exit={code}",
                flush=True,
            )
            if code:
                failed.append(name)
    if failed:
        print(f"failed: {failed}")


def pyscn(args) -> None:
    todo = []
    for label in [REFERENCE, *(lbl for lbl, _ in args.round)]:
        out_dir = args.work / "pyscn" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted((args.work / "src" / label).iterdir()):
            out = out_dir / f"{src.name}.json"
            if any(src.rglob("*.py")) and (args.force or not out.exists()):
                todo.append((src, out))
    print(f"{len(todo)} analyses")
    for src, out in todo:
        res = subprocess.run(
            ["uvx", PYSCN, "analyze", "--json", "--no-open", "--output", str(out), str(src)],
            capture_output=True, text=True,
        )  # fmt: skip
        if res.returncode or not out.exists():
            print(
                f"{out.parent.name}/{src.name} exit={res.returncode} {res.stderr[-300:]}"
            )


# ── analyze ───────────────────────────────────────────────────────────────────


def metrics(arcan_dir: Path) -> dict:
    """Size, coupling and smell metrics of one Arcan output."""
    comp = pd.read_csv(arcan_dir / "component-metrics.csv")
    files = comp[comp["ComponentType"] == "IMPL_FILE"]
    classes = comp[comp["ComponentType"] == "CLASS"]
    # The root container (the staged directory itself) is not a package.
    pkgs = comp[(comp["ComponentType"] == "PACKAGE") & comp["filePathRelative"].notna()]
    loc = files["LinesOfCode"].sum()
    kloc = loc / 1000 if loc else np.nan
    row = {
        "loc": loc,
        "files": len(files),
        "classes": len(classes),
        "packages": len(pkgs),
        "file_fan_out": files["FanOut"].mean(),
        "file_fan_in": files["FanIn"].mean(),
        "pkg_instability": pkgs["InstabilityMetric"].mean() if len(pkgs) else np.nan,
    }
    path = arcan_dir / "smell-characteristics.csv"
    smells = (
        pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    )
    if smells.empty:
        smells = pd.DataFrame(
            columns=["smellType", "AffectedComponentType", "ATDI", "Size"]
        )
    for smell, short in SMELLS.items():
        row[f"{short}_smells"] = int((smells["smellType"] == smell).sum())
    cd = smells[smells["smellType"] == "cyclicDep"]
    row["cd_unit"] = int((cd["AffectedComponentType"] == "UNIT").sum())
    row["cd_container"] = int((cd["AffectedComponentType"] == "CONTAINER").sum())
    row["cd_mean_size"] = cd["Size"].mean() if len(cd) else 0.0
    row["smells"] = len(smells)
    row["atdi"] = smells["ATDI"].sum()
    row["smells_per_kloc"] = row["smells"] / kloc
    row["atdi_per_kloc"] = row["atdi"] / kloc
    row["cd_per_kloc"] = row["cd_smells"] / kloc
    return row


PYSCN_OUTCOMES = [
    "health_score", "avg_complexity", "avg_cognitive", "high_complexity_share",
    "duplication_pct", "dead_code", "avg_cbo", "high_cbo_classes", "avg_lcom",
    "modules_in_cycles_share", "deps_max_depth", "main_seq_deviation",
    "modularity", "functions", "modules",
]  # fmt: skip


def pyscn_metrics(path: Path) -> dict:
    s = json.loads(path.read_text())["summary"]
    functions = s["total_functions"]
    modules = s["deps_total_modules"]
    return {
        "health_score": s["health_score"],
        "avg_complexity": s["average_complexity"],
        "avg_cognitive": s["average_cognitive_complexity"],
        "high_complexity_share": s["high_complexity_count"] / functions
        if functions
        else np.nan,
        "duplication_pct": s["code_duplication_percentage"],
        "dead_code": s["dead_code_count"],
        "avg_cbo": s["average_coupling"],
        "high_cbo_classes": s["high_coupling_classes"],
        "avg_lcom": s["average_lcom"],
        "modules_in_cycles_share": s["deps_modules_in_cycles"] / modules
        if modules
        else np.nan,
        "deps_max_depth": s["deps_max_depth"],
        "main_seq_deviation": s["deps_main_sequence_deviation"],
        "modularity": s["community_modularity"],
        "functions": functions,
        "modules": modules,
    }


OUTCOMES = [
    "smells_per_kloc", "atdi_per_kloc", "cd_per_kloc", "cd_unit", "cd_container",
    "cd_mean_size", "hl_smells", "gc_smells", "ud_smells", "smells", "atdi",
    "loc", "files", "classes", "packages", "file_fan_out", "pkg_instability",
]  # fmt: skip


def collect(args) -> pd.DataFrame:
    rows = []
    for label in [REFERENCE, *(lbl for lbl, _ in args.round)]:
        base = args.work / args.tool / label
        if args.tool == "arcan":
            base = base / "arcanOutput"
        for src in sorted((args.work / "src" / label).iterdir()):
            parts = src.name.split("__")
            name, condition, rep = (
                (parts[0], REFERENCE, 0)
                if label == REFERENCE
                else (parts[0], parts[1], int(parts[2]))
            )
            if name in args.exclude:
                continue
            row = {
                "round": label,
                "name": name,
                "condition": condition,
                "replicate": rep,
            }
            if args.tool == "arcan":
                out = base / src.name
                if (out / "component-metrics.csv").exists():
                    row.update(metrics(out))
            elif (base / f"{src.name}.json").exists():
                row.update(pyscn_metrics(base / f"{src.name}.json"))
            rows.append(row)
    return pd.DataFrame(rows)


def paired(a: pd.Series, b: pd.Series, seed: int = 0) -> dict:
    d = (b - a).dropna()
    rng = np.random.default_rng(seed)
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)] if len(d) else [np.nan]
    p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
    return {
        "n": len(d),
        "diff": d.mean(),
        "ci_low": np.percentile(boot, 2.5),
        "ci_high": np.percentile(boot, 97.5),
        "higher": int((d > 0).sum()),
        "lower": int((d < 0).sum()),
        "p": p,
    }


def analyze(args) -> None:
    frame = collect(args)
    outcomes, key = (
        (OUTCOMES, "loc") if args.tool == "arcan" else (PYSCN_OUTCOMES, "functions")
    )
    frame = frame.reindex(columns=[*frame.columns[:4], *outcomes])
    missing = frame[frame[key].isna()]
    if len(missing):
        print(f"no {args.tool} output for {len(missing)} trees:")
        print(
            missing[["round", "name", "condition", "replicate"]].to_string(index=False)
        )
    if args.csv:
        frame.to_csv(args.csv, index=False)
    frame = frame.dropna(subset=[key])
    ref = frame[frame["condition"] == REFERENCE].set_index("name")[outcomes]
    agents = frame[frame["condition"] != REFERENCE]
    pd.set_option("display.width", 240)

    def per_library(df: pd.DataFrame) -> pd.DataFrame:
        # library mean over replicates (and over rounds, when pooled)
        return df.groupby(["name", "condition"])[outcomes].mean()

    scopes = [("pooled", agents)] + [
        (lbl, agents[agents["round"] == lbl]) for lbl, _ in args.round
    ]
    for scope, df in scopes:
        lib = per_library(df)
        intent = lib.xs("intent_only", level="condition")
        full = lib.xs("req_arch", level="condition")
        names = intent.index.intersection(full.index).intersection(ref.index)
        print(f"\n== {scope}: {len(names)} libraries, library means ==")
        means = pd.DataFrame(
            {
                "reference": ref.loc[names].mean(),
                "intent": intent.loc[names].mean(),
                "full": full.loc[names].mean(),
            }
        )
        print(means.round(2).to_string())
        tests = []
        for m in outcomes:
            for label, a, b in (
                ("full-intent", intent[m], full[m]),
                ("intent-ref", ref[m], intent[m]),
                ("full-ref", ref[m], full[m]),
            ):
                tests.append(
                    {
                        "metric": m,
                        "contrast": label,
                        **paired(a.loc[names], b.loc[names]),
                    }
                )
            # Which condition is closer to the reference (absolute distance)?
            dist_i = (intent[m] - ref[m]).abs().loc[names]
            dist_f = (full[m] - ref[m]).abs().loc[names]
            tests.append(
                {
                    "metric": m,
                    "contrast": "|full-ref|-|intent-ref|",
                    **paired(dist_i, dist_f),
                }
            )
        print(f"\n== {scope}: paired over libraries (Wilcoxon, bootstrap 95% CI) ==")
        print(pd.DataFrame(tests).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=("stage", "run", "pyscn", "analyze"))
    parser.add_argument("--tool", choices=("arcan", "pyscn"), default="arcan")
    parser.add_argument(
        "--round", nargs=2, action="append", metavar=("LABEL", "OUT"),
        help="Benchmark round: label and output folder (default: Gemma and Q35)",
    )  # fmt: skip
    parser.add_argument(
        "--work", type=Path, default=ROOT / "data/interim/arcan_commit0"
    )
    parser.add_argument("--scaffold", default="contract")
    parser.add_argument(
        "--instances",
        type=Path,
        default=ROOT / "data/raw/commit0/commit0_instances.jsonl",
    )
    parser.add_argument("--exclude", nargs="+", default=[], metavar="NAME")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per analysis")
    parser.add_argument("--java-opts", default="-Xmx4g -XX:+ExitOnOutOfMemoryError")
    parser.add_argument("--force", action="store_true", help="Redo existing outputs")
    parser.add_argument("--csv", type=Path, default=None, help="Write per-run metrics")
    args = parser.parse_args()
    args.round = args.round or ROUNDS
    args.work = args.work.resolve()

    if args.step == "stage":
        instances = {i["name"]: i for i in commit0.load_instances(args.instances)}
        stage(args, instances)
    elif args.step == "run":
        run(args)
    elif args.step == "pyscn":
        pyscn(args)
    else:
        analyze(args)


if __name__ == "__main__":
    sys.exit(main())
