#!/usr/bin/env python3
"""
Compare two TSV files (repo<TAB>language) and list repos whose normalized language
differs between file1 and file2.

Normalization:
- C++ / Cpp / etc -> CPP
- C# -> CSHARP
- Python -> PYTHON
- Java -> JAVA
- Header handling: e.g. "C Header" -> "C" (keep only the first token)
- Only consider languages in: CPP, CSHARP, C, PYTHON, JAVA
- Ignore rows whose normalized language is not in that list (e.g., PowerShell)
- Output only repos that exist in BOTH files and have different normalized language.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET_LANGS = {"CPP", "CSHARP", "C", "PYTHON", "JAVA"}


def normalize_language(raw: str) -> str | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None

    # Keep only "main language" (first token) e.g. "C Header" -> "C"
    main = s.split()[0]

    # Canonicalize common variants
    main_upper = main.upper()

    # Map symbols/variants
    if main in {"C++", "c++"} or main_upper in {"C++"}:
        norm = "CPP"
    elif main in {"C#", "c#"} or main_upper in {"C#"}:
        norm = "CSHARP"
    elif main_upper == "PYTHON":
        norm = "PYTHON"
    elif main_upper == "JAVA":
        norm = "JAVA"
    elif main_upper == "C":
        norm = "C"
    elif main_upper == "CPP":
        norm = "CPP"
    elif main_upper == "CSHARP":
        norm = "CSHARP"
    else:
        # Unknown/unwanted language
        return None

    return norm if norm in TARGET_LANGS else None


def read_tsv(path: Path) -> dict[str, str]:
    """
    Reads TSV with 2+ columns:
      repo<TAB>language[<TAB>...]
    Returns mapping repo -> normalized_language
    If a repo appears multiple times, the last valid (kept) row wins.
    """
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip("\n\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                # allow space-separated fallback
                parts = line.split()
                if len(parts) < 2:
                    print(
                        f"Warning: {path}:{i} could not parse line (need repo + language): {line!r}",
                        file=sys.stderr,
                    )
                    continue

            repo = parts[0].strip()
            lang_raw = parts[1].strip()
            if not repo:
                continue

            lang_norm = normalize_language(lang_raw)
            if lang_norm is None:
                continue  # ignore non-target languages
            out[repo] = lang_norm
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="List repos whose language differs between TSV1 and TSV2 after normalization."
    )
    ap.add_argument("file1", type=Path, help="TSV file 1: repo<TAB>language")
    ap.add_argument("file2", type=Path, help="TSV file 2: repo<TAB>language")
    ap.add_argument(
        "--show-langs",
        action="store_true",
        help="Also print file1_lang and file2_lang columns.",
    )
    args = ap.parse_args()

    m1 = read_tsv(args.file1)
    m2 = read_tsv(args.file2)

    common = sorted(set(m1) & set(m2))
    diffs = [r for r in common if m1[r] != m2[r]]

    if args.show_langs:
        print("repo\tfile1_lang\tfile2_lang")
        for r in diffs:
            print(f"{r}\t{m1[r]}\t{m2[r]}")
    else:
        for r in diffs:
            print(r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
