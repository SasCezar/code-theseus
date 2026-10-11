#!/usr/bin/env bash
# Batch-render C4 markdown docs to PDF.
# All processing (strip_think, render, fix_mermaid, retry) runs inside
# a single Docker container — one startup cost for all files.
#
# Usage:
#   ./scripts/bash/render_c4_pdf.sh [docs_dir] [out_dir]
#
# Environment:
#   DOCKER_IMAGE   md2pdf image name  (default: md2pdf)
#   PARALLEL_JOBS  concurrent renders (default: 4)

set -euo pipefail

DOCS_DIR="$(realpath "${1:-data/generated_docs/c4}")"
OUT_DIR="$(realpath  "${2:-data/generated_docs/c4_pdf}")"
IMAGE="${DOCKER_IMAGE:-md2pdf}"
JOBS="${PARALLEL_JOBS:-4}"

mkdir -p "$OUT_DIR"

echo "Image:   $IMAGE"
echo "Docs:    $DOCS_DIR"
echo "Output:  $OUT_DIR"
echo "Jobs:    $JOBS"
echo ""

docker run --rm -t \
    --platform linux/amd64 \
    -v "$DOCS_DIR:/docs:ro" \
    -v "$OUT_DIR:/output" \
    -e JOBS="$JOBS" \
    --entrypoint render_batch \
    "$IMAGE" /docs /output
