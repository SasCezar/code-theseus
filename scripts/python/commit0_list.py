#!/usr/bin/env python3
"""Export the Commit0 benchmark's 54 libraries as reverse-se pipeline input.

Source of truth is the official HF dataset `wentingzhao/commit0_combined`
(split `test`, one row per library). Each row points at a `commit-0/<name>`
GitHub fork with two relevant commits:
  - `reference_commit`: the full original implementation -- what the doc
    pipeline (Arcan -> context -> intent/req/c4) must analyze;
  - `base_commit`: Commit0's stubbed starter repo -- NOT what we analyze.
The forks' default branch is the stubbed starter, so a plain depth-1 clone
(prefetch_repo.sh) would analyze stubs. Use prefetch_commit0.sh instead,
which checks out `reference_commit`.

Writes:
  <out_dir>/commit0_list.tsv        url<TAB>PYTHON, the TSV format consumed by
                                    prefetch/arcan/context (Makefile `TSV=`)
  <out_dir>/commit0_instances.jsonl full per-library metadata (commits,
                                    src/test dirs, setup, lite-split flag)

Usage:
    uv run scripts/python/commit0_list.py [--out-dir data/raw/commit0] [--lite]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=wentingzhao/commit0_combined&config=default&split=test"
    "&offset={offset}&length={length}"
)

# commit0/harness/constants.py SPLIT_LITE (commit-0/commit0 @ main); the HF
# dataset itself carries no split flag.
SPLIT_LITE = {
    "tinydb",
    "simpy",
    "deprecated",
    "wcwidth",
    "voluptuous",
    "cachetools",
    "imapclient",
    "marshmallow",
    "jinja",
    "cookiecutter",
    "portalocker",
    "parsel",
    "pyjwt",
    "chardet",
    "babel",
    "minitorch",
}


def fetch_rows(page: int = 50) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        with urllib.request.urlopen(
            HF_ROWS_URL.format(offset=offset, length=page)
        ) as resp:
            data = json.load(resp)
        rows += [r["row"] for r in data["rows"]]
        offset += page
        if offset >= data["num_rows_total"]:
            return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/commit0"))
    parser.add_argument(
        "--lite", action="store_true", help="only the 16-library lite split"
    )
    args = parser.parse_args()

    rows = fetch_rows()
    records = []
    for r in rows:
        name = r["repo"].split("/", 1)[1]
        records.append(
            {
                "project_id": r["repo"].replace(
                    "/", "|"
                ),  # matches prefetch's user|repo dir
                "name": name,
                "url": f"https://github.com/{r['repo']}",
                "original_repo": r["original_repo"],
                "reference_commit": r["reference_commit"],
                "base_commit": r["base_commit"],
                "src_dir": r["src_dir"],
                "test": r["test"],
                "setup": r["setup"],
                "lite": name in SPLIT_LITE,
            }
        )
    if args.lite:
        records = [rec for rec in records if rec["lite"]]
    records.sort(key=lambda rec: (not rec["lite"], rec["name"].lower()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "commit0_list.tsv").open("w") as f:
        for rec in records:
            f.write(f"{rec['url']}\tPYTHON\n")
    with (args.out_dir / "commit0_instances.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(
        f"{len(records)} libraries ({sum(r['lite'] for r in records)} lite) -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
