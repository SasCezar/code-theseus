#!/usr/bin/env bash
# Clone each Commit0 fork and check out its *reference* commit (the full
# original implementation), for Arcan/context/doc generation. Unlike
# prefetch_repo.sh's depth-1 default-branch clone -- which on a commit-0/*
# fork yields Commit0's stubbed starter repo -- this pins reference_commit.
# The checkout is put on a local `main` branch: on a detached HEAD Arcan's
# LastVersionEnumerationStrategy discovers 0 versions and writes no output.
# reference_commit may be an annotated tag object; HEAD is its peeled commit.
#
# Usage:
#   ./scripts/bash/prefetch_commit0.sh data/raw/commit0/commit0_instances.jsonl
#   (produced by scripts/python/commit0_list.py)
set -euo pipefail

JSONL="${1:-}"
[[ -n "$JSONL" && -f "$JSONL" ]] || { echo "Usage: $0 commit0_instances.jsonl" >&2; exit 2; }

DEST_BASE="${DEST_BASE:-$(pwd)/data/repository}"
mkdir -p "$DEST_BASE"

python3 -c '
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(r["url"], r["project_id"], r["reference_commit"])
' "$JSONL" | while read -r url pid ref; do
  dir="${DEST_BASE}/${pid}"
  if [[ -d "$dir/.git" ]] && [[ "$(git -C "$dir" rev-parse HEAD)" == "$(git -C "$dir" rev-parse -q --verify "${ref}^{commit}" 2>/dev/null)" ]]; then
    echo "[SKIP] $pid already at ${ref:0:8}"
    continue
  fi
  rm -rf "$dir"
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  if git -C "$dir" fetch -q --depth 1 origin "$ref" && git -C "$dir" checkout -q -B main FETCH_HEAD; then
    echo "[OK]   $pid @ ${ref:0:8}"
  else
    echo "[FAIL] $pid @ ${ref:0:8}" >&2
    rm -rf "$dir"
  fi
done
