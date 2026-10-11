#!/usr/bin/env python3
"""Export repo info, Arcan graphs and generated docs as a Hugging Face dataset.

Writes two Parquet configs under --out:

    repos/train.parquet      one row per repository: GitHub info, analysed
                             commit, and the package-level Arcan graph the
                             generators were prompted with.
    documents/train.parquet  one row per (repo, generator): the intent,
                             requirements and architecture (C4) documents,
                             their finish reasons and, when the generator
                             emits one, its reasoning trace.

Only data we produced is exported: no source code (`context_files`) and no
rendered prompts, so the release carries no third-party file contents.

Usage:
    uv run --with pyarrow scripts/python/export_hf.py --repo <owner>/<name> \
        [--code-url URL] [--max-repos N]
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DATASETS = {
    "main": Path("data/out/reverse-se_v0.6_p10-95_ctx95_dataset.jsonl"),
    "commit0": Path("data/out/commit0_v0.6_ctx95_dataset.jsonl"),
}
DOCS_DIR = Path("data/out/llm_results")

# model_slug dir -> (generator name, HF checkpoint)
GENERATORS = {
    "hosted_vllm--qwen-3.5-122b": ("qwen3.5-122b-a10b", "Qwen/Qwen3.5-122B-A10B-FP8"),
    "hosted_vllm--gemma-4-31b": ("gemma-4-31b-it", "google/gemma-4-31B-it"),
}
CONTEXT_BUDGET = "ctx95"
# on-disk task dir -> exported column name
ARTEFACTS = {"intent": "intent", "req": "requirements", "c4": "architecture"}

NODE = pa.struct(
    [
        ("vertex_id", pa.int64()),
        ("name", pa.string()),
        ("component_type", pa.string()),
        ("fan_in", pa.int64()),
        ("fan_out", pa.int64()),
        ("lines_of_code", pa.int64()),
        ("pagerank", pa.float64()),
    ]
)
EDGE = pa.struct(
    [("source", pa.int64()), ("target", pa.int64()), ("weight", pa.int64())]
)
REPOS_SCHEMA = pa.schema(
    [
        ("repo_id", pa.string()),
        ("branch", pa.string()),
        ("url", pa.string()),
        ("commit_sha", pa.string()),
        ("language", pa.string()),
        ("description", pa.string()),
        ("stars", pa.int64()),
        (
            "arcan_graph",
            pa.struct([("nodes", pa.list_(NODE)), ("edges", pa.list_(EDGE))]),
        ),
    ]
)
DOCS_SCHEMA = pa.schema(
    [("repo_id", pa.string()), ("generator", pa.string()), ("checkpoint", pa.string())]
    + [(col, pa.string()) for col in ARTEFACTS.values()]
    + [(f"reasoning_{col}", pa.string()) for col in ARTEFACTS.values()]
)


def repo_row(rec: dict, subset: str) -> dict:
    info = rec["info"]
    graph = rec["package_only_graph"]
    return {
        "repo_id": rec["repo_id"],
        "branch": subset,
        "url": info["url"],
        "commit_sha": info["hash"],
        "language": info["language"],
        "description": info.get("description"),
        "stars": info.get("stargazers_count"),
        "arcan_graph": {
            "nodes": [{k: n[k] for k in NODE.names} for n in graph["nodes"]],
            "edges": [{k: e[k] for k in EDGE.names} for e in graph["edges"]],
        },
    }


def export_repos(out: Path, max_repos: int | None = None) -> set[str]:
    """Stream the dataset JSONLs into repos/train.parquet; return kept ids.

    `max_repos` caps each subset (for a small draft release)."""
    path = out / "repos" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    kept: set[str] = set()
    batch: list[dict] = []
    with pq.ParquetWriter(path, REPOS_SCHEMA, compression="zstd") as writer:
        for subset, src in DATASETS.items():
            n = 0
            with src.open(encoding="utf-8") as f:
                for line in tqdm(f, desc=f"repos/{subset}"):
                    if max_repos is not None and n >= max_repos:
                        break
                    rec = json.loads(line)
                    rid = rec["repo_id"]
                    if rid in kept:
                        continue
                    kept.add(rid)
                    n += 1
                    batch.append(repo_row(rec, subset))
                    if len(batch) == 500:
                        writer.write_table(pa.Table.from_pylist(batch, REPOS_SCHEMA))
                        batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, REPOS_SCHEMA))
    return kept


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def export_documents(out: Path, repo_ids: set[str]) -> int:
    """Write documents/train.parquet; return how many docs were dropped.

    Truncated (`finish_reason != "stop"`) or empty outputs are stored as null
    so no partial document ends up in training data."""
    rows = []
    dropped = 0
    for slug, (generator, checkpoint) in GENERATORS.items():
        base = DOCS_DIR / slug / CONTEXT_BUDGET
        for rid in tqdm(sorted(repo_ids), desc=f"documents/{generator}"):
            row = {"repo_id": rid, "generator": generator, "checkpoint": checkpoint}
            for task, col in ARTEFACTS.items():
                d = base / task
                text = read_text(d / f"{rid}.md")
                meta = read_text(d / f"{rid}.meta.json")
                finish = json.loads(meta).get("finish_reason") if meta else None
                if text is not None and (finish != "stop" or not text.strip()):
                    dropped += 1
                    text = None
                row[col] = text
                row[f"reasoning_{col}"] = (
                    read_text(d / f"{rid}.reasoning.txt") if text else None
                )
            if any(row[col] is not None for col in ARTEFACTS.values()):
                rows.append(row)
    path = out / "documents" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, DOCS_SCHEMA), path, compression="zstd")
    return dropped


CARD = """---
license: cc-by-4.0
pretty_name: code-theseus
task_categories:
- text-generation
language:
- en
size_categories:
- 10K<n<100K
tags:
- software-engineering
- requirements-engineering
- software-architecture
- reverse-engineering
- c4-model
- code-generation
- coding-agents
- llm
- synthetic
configs:
- config_name: documents
  data_files: documents/train.parquet
  default: true
- config_name: repos
  data_files: repos/train.parquet
---

<div align="center">

# 🚢 code-theseus

**The design behind 10,000+ open-source repositories: intent, requirements
and architecture, aligned with the code, ready for training coding agents.**

| 📦 **{n_repos}** | 📄 **{n_docs}** | 🌐 **4** | 🤖 **{n_generators}** |
|:---:|:---:|:---:|:---:|
| repositories | document sets | languages: Java, Python, C/C++, C# | generator LLMs |

{links}

</div>

## 🧭 Why this dataset

Coding agents increasingly design before they code, but almost no data
connects code to the requirements and architecture behind it: real projects
rarely keep these documents, and when they do they are outdated or private.
code-theseus recovers them from the source code, at scale.

## 📦 What's inside

Every repository comes with three aligned documents, generated from its
dependency graph and key project files:

| | Document | What it captures |
|---|---|---|
| 🎯 | **`intent`** | The "initial prompt": mission, problem, capabilities, user workflows, quality expectations |
| 📋 | **`requirements`** | `REQ-FUNC-*` / `REQ-NFR-*` items with statement, rationale, acceptance criteria, verification method and priority |
| 🏗️ | **`architecture`** | A C4 design (context, container, component) with Mermaid C4 diagrams |

…plus the **package-level dependency graph** each document was generated
from, and a link to the exact commit it describes. No source code is
redistributed: check out `url` at `commit_sha` to recover it.

## 🔁 Does it carry the design?

In the paper's Ship of Theseus test, a coding agent rebuilt
[Commit0](https://github.com/commit-0/commit0) libraries from these
documents alone. With requirements and architecture, it built a module
structure **significantly closer to the original** than with intent alone
(module F1 0.54 → 0.63), at the same test pass rate.

## 🏢 Use it for

- **Fine-tuning** models that write specs from code (graph → intent,
  requirements, architecture) or code from specs.
- **Spec-driven coding agents.** Realistic requirements and architecture
  briefs as agent inputs.
- **Evaluating** the intermediate steps of multi-agent development, not just
  the final code.
- **Research** on requirements engineering, software architecture recovery
  and traceability.

## 🚀 Quick start

```python
from datasets import load_dataset

docs = load_dataset("{repo}", "documents", split="train")
repos = load_dataset("{repo}", "repos", split="train")
graphs = dict(zip(repos["repo_id"], repos["arcan_graph"]))
graph = graphs[docs[0]["repo_id"]]  # input graph for the first document
```

## 🗂️ Configs

### `documents` ({n_docs} rows): one row per (repository, generator)

| column | description |
|---|---|
| `repo_id` | `owner\\|repo`; join key to `repos` |
| `generator` | {generators} |
| `checkpoint` | Hugging Face checkpoint served with vLLM |
| `intent`, `requirements`, `architecture` | generated Markdown; null if missing, truncated or empty ({n_dropped} docs) |
| `reasoning_*` | the generator's thinking trace (Qwen only, null otherwise) |

### `repos` ({n_repos} rows): one row per repository

| column | description |
|---|---|
| `repo_id` | `owner\\|repo` |
| `branch` | `main` (GitHub corpus) or `commit0` ([Commit0](https://github.com/commit-0/commit0) libraries) |
| `url` | GitHub URL |
| `commit_sha` | analysed commit; documents describe the repository at this commit |
| `language`, `description`, `stars` | GitHub metadata at collection time |
| `arcan_graph` | package-level dependency graph from [Arcan](https://www.arcan.tech/): `nodes` (`vertex_id`, `name`, `component_type`, `fan_in`, `fan_out`, `lines_of_code`, `pagerank`) and weighted `edges` (`source`/`target` are `vertex_id`s) |

## 📜 License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This covers the
generated documents and dependency graphs. The original repositories keep
their own licenses.
"""


def write_card(out: Path, repo: str, code_url: str | None, n_dropped: int) -> None:
    docs = pq.read_table(out / "documents" / "train.parquet", columns=["generator"])
    gens = sorted(set(docs.column("generator").to_pylist()))
    n_repos = pq.read_metadata(out / "repos" / "train.parquet").num_rows
    (out / "README.md").write_text(
        CARD.replace("{n_docs}", f"{docs.num_rows:,}")
        .replace("{n_repos}", f"{n_repos:,}")
        .replace("{n_dropped}", f"{n_dropped:,}")
        .replace("{generators}", " / ".join(f"`{g}`" for g in gens))
        .replace("{n_generators}", str(len(gens)))
        .replace("{links}", f"[💻 Code]({code_url})" if code_url else "")
        .replace("{repo}", repo),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/out/hf_export"))
    ap.add_argument("--repo", required=True, help="HF dataset id, owner/name")
    ap.add_argument("--code-url", help="generation code URL to cite in the card")
    ap.add_argument("--max-repos", type=int, help="cap per subset, for a draft")
    args = ap.parse_args()

    repo_ids = export_repos(args.out, args.max_repos)
    dropped = export_documents(args.out, repo_ids)
    write_card(args.out, args.repo, args.code_url, dropped)


if __name__ == "__main__":
    main()
