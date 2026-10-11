#!/usr/bin/env python3

import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"\bname=([^\s]+)")


def extract_status_and_name(line: str):
    line = line.strip()
    if not line.startswith("["):
        return None, None

    close = line.find("]")
    if close == -1:
        return None, None

    status = line[1:close]
    m = NAME_RE.search(line)
    if not m:
        return None, None

    return status, m.group(1)


def find_started_not_ended(logfile: Path):
    started = set()
    ended = set()

    with logfile.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            status, name = extract_status_and_name(line)
            if not status or not name:
                continue

            if status == "START":
                started.add(name)
            elif status == "END":
                ended.add(name)

    # Only those that started but never ended
    return sorted(started - ended)


def main():
    if len(sys.argv) != 2:
        print("Usage: python find_unfinished.py <logfile>")
        sys.exit(1)

    logfile = Path(sys.argv[1])
    if not logfile.exists():
        print(f"File not found: {logfile}")
        sys.exit(1)

    unfinished = find_started_not_ended(logfile)

    print(f"Projects with [START] but no [END]: {len(unfinished)}")
    print()
    for name in unfinished:
        print(name)


if __name__ == "__main__":
    main()
