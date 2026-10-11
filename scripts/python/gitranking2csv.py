import csv
import json
from pathlib import Path

from utils import ROOT_DIR

ALLOWED_LANGUAGES = {"java", "c++", "c", "python"}


def normalize_language(lang):
    if lang is None:
        return None
    return lang.strip().lower()


def main(inp_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        inp_path.open("r", encoding="utf-8") as f_in,
        out_path.open("w", encoding="utf-8", newline="") as f_out,
    ):
        writer = csv.DictWriter(
            f_out, fieldnames=["name", "url", "description", "language"]
        )
        writer.writeheader()

        for lineno, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno}") from e

            lang = obj.get("language")
            if normalize_language(lang) not in ALLOWED_LANGUAGES:
                continue

            if obj.get("archived"):
                continue

            row = {
                "name": obj.get("full_name") or obj.get("name") or "",
                "url": obj.get("html_url"),
                "description": obj.get("description"),
                "language": lang,
            }
            writer.writerow(row)

    print(f"CSV written to: {out_path}")


if __name__ == "__main__":
    inp_path = Path(ROOT_DIR / "data/raw/repo-lists/github_projects.json")
    out_path = Path(ROOT_DIR / "data/raw/github_projects.csv")
    main(inp_path, out_path)
