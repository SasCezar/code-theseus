import csv
import json
from pathlib import Path

import pandas as pd
import tiktoken
from tiktoken import Encoding

INPUT_FILE = "data/interim/repo_context_files_all.jsonl"
ANALYZED_FILE = "data/interim/dataset_stats.csv"
OUTPUT_FILE = "data/interim/repo_context_files_analyzed_only_stats.csv"

MODEL = "gpt-5-mini"

enc: Encoding = tiktoken.encoding_for_model(MODEL)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text, disallowed_special=()))


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    analyzed_path = Path(ANALYZED_FILE)
    analyzed_repos: set[str] = set(pd.read_csv(analyzed_path)["repo"])
    print(len(analyzed_repos))
    with (
        input_path.open("r", encoding="utf-8") as f_in,
        output_path.open("w", encoding="utf-8", newline="") as f_out,
    ):
        writer = csv.writer(f_out)
        writer.writerow(["repo", "language", "path", "category", "token_n"])

        for line in f_in:
            record = json.loads(line)

            repo = record.get("repo")
            repo = f"https://github.com/{repo.replace('|', '/')}"
            if repo not in analyzed_repos:
                continue
            language = record.get("language")
            documents = record.get("documents", [])

            for doc in documents:
                path = doc.get("path")
                category = doc.get("category")
                content = doc.get("content", "")

                token_n = count_tokens(content)

                writer.writerow([repo, language, path, category, token_n])

    print(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    main()
