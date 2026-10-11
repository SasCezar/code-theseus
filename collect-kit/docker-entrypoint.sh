#!/bin/sh
# docker-entrypoint.sh — runs inside the combined collector image.
# Executes Arcan, then runs collect_repo.py.
set -eu

REPO_NAME="${REPO_NAME:?REPO_NAME env var is required (the folder name of the repo)}"
LANGUAGE="${LANGUAGE:?LANGUAGE env var is required (C | CPP | JAVA | PYTHON | CSHARP)}"
EXCLUDE="${EXCLUDE_CATEGORIES:-}"
MAX_BYTES="${MAX_BYTES:-300000}"

REPO_INPUT="/data/repos/$REPO_NAME"
ARCAN_OUT_DIR="/data/arcan_out"
ARCAN_REPO_OUT="$ARCAN_OUT_DIR/arcanOutput/$REPO_NAME"
OUTPUT_FILE="/data/output/$REPO_NAME.jsonl"

echo "━━━ Step 1/2: Running Arcan ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Repo:     $REPO_INPUT"
echo "  Language: $LANGUAGE"
echo ""

# Arcan's ENTRYPOINT is /arcan/arcan.sh (WORKDIR /arcan in the base image).
/arcan/arcan.sh analyze \
    -i "$REPO_INPUT" \
    -o "$ARCAN_OUT_DIR" \
    -l "$LANGUAGE" \
    --all \
    output.writeDependencyGraph=true \
    output.writeMembership=true \
    output.writeDependency=true

echo ""
echo "━━━ Step 2/2: Collecting data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EXTRA=""
if [ -n "$EXCLUDE" ]; then
    EXTRA="--exclude-categories $EXCLUDE"
fi

# shellcheck disable=SC2086
python3 /opt/collect_repo.py \
    --repo-path "$REPO_INPUT" \
    --arcan-output "$ARCAN_REPO_OUT" \
    --language "$LANGUAGE" \
    --output "$OUTPUT_FILE" \
    --max-bytes "$MAX_BYTES" \
    $EXTRA

echo ""
echo "━━━ Done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Output: $OUTPUT_FILE"
