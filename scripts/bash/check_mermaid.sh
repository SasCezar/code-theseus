#!/usr/bin/env bash
# Mermaid diagram-compilation check: walks <results_dir>/<model>/<context>/c4/*.md,
# extracts every ```mermaid block, and tries to render each with mermaid.render()
# in a single headless Chromium session (no PDF output — syntax validity only).
# Writes a per-diagram TSV (model, context, task, repo_id, diagram_idx, status, error).
#
# Usage:
#   ./scripts/bash/check_mermaid.sh [results_dir] [out_dir]
#
# Environment:
#   DOCKER_IMAGE   md2pdf image name  (default: md2pdf; build with:
#                  docker build --platform linux/amd64 -t md2pdf docker/md2pdf)
#   MERMAID_JOBS   concurrent renderer tabs (default: 8)

set -euo pipefail

RESULTS_DIR="$(realpath "${1:-data/out/llm_results}")"
OUT_DIR="$(realpath "${2:-data/out/doc_checks}")"
IMAGE="${DOCKER_IMAGE:-md2pdf}"
JOBS="${MERMAID_JOBS:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$OUT_DIR"

echo "Image:   $IMAGE"
echo "Results: $RESULTS_DIR"
echo "Output:  $OUT_DIR/mermaid_check.tsv"
echo "Jobs:    $JOBS"
echo ""

docker run --rm -t \
    --platform linux/amd64 \
    -v "$RESULTS_DIR:/docs:ro" \
    -v "$SCRIPT_DIR/docker/md2pdf/check_mermaid.js:/usr/local/bin/check_mermaid.js:ro" \
    -v "$OUT_DIR:/output" \
    -e MERMAID_JOBS="$JOBS" \
    --entrypoint node \
    "$IMAGE" /usr/local/bin/check_mermaid.js /docs /output/mermaid_check.tsv
