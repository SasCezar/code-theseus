"""Share of a repository's code units that its generated documents name.

Units come from the raw Arcan graph in the dataset: Python modules (as in
module F1 of the agent benchmark), C/C++ files and Java/C# types. Test code
is left out. A unit counts as named when the requirements or architecture
document contains its full name or path, a unique file name (`utils.c`,
`client.py`), a unique CamelCase type name, or a C4 element alias equal to,
extending or shortening its last segment (5+ characters) -- the matching of
scripts/python/structure_commit0.doc_modules, extended to the other
languages. Only explicit names count, so the share is a lower bound on what
the documents describe.
"""

import ast
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

UNIT_TYPES = {
    "Python": {"IMPL_FILE"},
    "C": {"IMPL_FILE", "HEADER"},
    "C++": {"IMPL_FILE", "HEADER"},
    "Java": {"CONCRETE_CLASS", "INTERFACE", "ENUM", "ABSTRACT_CLASS", "ANNOTATION"},
    "C#": {"CLASS", "INTERFACE", "ENUM"},
}
DOCS = ("req", "c4")
_TEST = re.compile(
    r"(^|[./_-])(tests?|testing|spec|specs)([./_-]|$)|Tests?$|Test[A-Z_]|^test_|_test\b",
    re.I,
)
_C_FILE = re.compile(r"\.(c|cc|cpp|cxx|h|hh|hpp|hxx)$")
_CAMEL = re.compile(r"[A-Z].*[a-z]|[a-z].*[A-Z]")
_C4_ALIAS = re.compile(r"\b(?:Container|Component|System)\w*\(\s*([A-Za-z_]\w*)\s*,")


def _parse(value):
    return value if isinstance(value, (dict, list)) else ast.literal_eval(value)


def units(graph: dict, language: str, repo_id: str) -> set[str]:
    """Non-test code units of a raw Arcan graph."""
    out = set()
    for node in graph["nodes"]:
        if node["component_type"] not in UNIT_TYPES.get(language, ()):
            continue
        name = node["name"].split("<")[0]  # generic Java/C# types
        if name.startswith(repo_id + "."):
            name = name[len(repo_id) + 1 :]
        if name and name != repo_id and not _TEST.search(name):
            out.add(name)
    return out


def _last(name: str) -> str:
    return name if _C_FILE.search(name) else name.rpartition(".")[2]


def named(text: str, names: set[str], python: bool) -> set[str]:
    lasts = Counter(_last(n).lower() for n in names)
    aliases = {a.lower() for a in _C4_ALIAS.findall(text)}
    found = set()
    for name in names:
        last = _last(name)
        stem = last.lower().rsplit(".", 1)[0]
        patterns = [rf"(?<![\w.]){re.escape(name)}(?!\w)"]
        if "." in name and not _C_FILE.search(name):
            patterns.append(rf"(?<!\w){re.escape(name.replace('.', '/'))}(?!\w)")
        if lasts[last.lower()] == 1 and len(stem) >= 3:
            if python:
                patterns.append(rf"(?<![\w/.]){re.escape(last)}\.py\b")
            elif "." in last or _CAMEL.search(last):
                patterns.append(rf"(?<!\w){re.escape(last)}(?!\w)")
        alias = any(
            a == stem
            or (len(stem) >= 5 and a.startswith(stem))
            or (len(a) >= 5 and stem.startswith(a))
            for a in aliases
        )
        if alias or any(re.search(p, text) for p in patterns):
            found.add(name)
    return found


def _repo(line: str, results_dir: Path, context: str, models: list[str]) -> list[dict]:
    record = json.loads(line)
    repo_id = record["repo_id"]
    language = _parse(record["info"]).get("language")
    names = units(_parse(record["original_graph"]), language, repo_id)
    rows = []
    for model in models:
        paths = [results_dir / model / context / doc / f"{repo_id}.md" for doc in DOCS]
        if not names or not all(p.is_file() for p in paths):
            continue
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in paths)
        hit = named(text, names, python=language == "Python")
        rows.append(
            {
                "model": model,
                "repo_id": repo_id,
                "language": language,
                "units": len(names),
                "units_named": len(hit),
                "units_named_pct": 100 * len(hit) / len(names),
            }
        )
    return rows


def unit_coverage(
    dataset: Path, results_dir: Path, context: str, models: list[str]
) -> pd.DataFrame:
    """One row per (model, repository): units in the graph and units named
    in its requirements and architecture documents."""
    with open(dataset, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    work = partial(_repo, results_dir=results_dir, context=context, models=models)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        return pd.DataFrame(
            [row for rows in pool.map(work, lines, chunksize=1) for row in rows]
        )
