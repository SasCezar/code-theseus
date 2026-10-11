#!/usr/bin/env bash
# collect.sh — Run Arcan + collect architecture/context data for a single repo.
# Works on Linux and macOS (including Apple Silicon M-series).
#
# Usage:
#   ./collect.sh --repo /path/to/myrepo --language JAVA --output ./myrepo.jsonl
#
# Full options: ./collect.sh --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCAN_IMAGE="ghcr.io/arcan-tech/arcan-2-cli-trial:latest"

# ── Defaults ──────────────────────────────────────────────────────────────────
REPO_PATH=""
LANGUAGE=""
OUTPUT_FILE=""
ARCAN_OUTPUT_DIR=""
KEEP_ARCAN_OUTPUT=0
EXCLUDE_CATEGORIES=""
MAX_BYTES=300000
PHASE_A_DEPTH=2
PHASE_A_MAX=20000
PHASE_B_DEPTH=4
PHASE_B_MAX=60000

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --repo PATH --language LANG --output FILE [options]

Required:
  --repo PATH             Path to the repository to analyse
  --language LANG         Language: C | CPP | JAVA | PYTHON | CSHARP
  --output FILE           Output JSONL file (e.g. ./myrepo.jsonl)

Optional:
  --arcan-output DIR      Directory to store Arcan's raw output
                          (default: <output-dir>/arcan_out/<repo-name>)
  --keep-arcan-output     Keep the Arcan output directory after collection
  --exclude-categories X  Comma-separated categories to omit from context files
                          Choices: readme, docker, build, dependencies
                          Example: --exclude-categories readme,docker
  --max-bytes N           Max bytes per context file (default: 300000)
  --help                  Show this help message

Examples:
  # Collect a Java project, skip the README
  ./collect.sh --repo ~/projects/myapp --language JAVA \\
               --output ~/data/myapp.jsonl \\
               --exclude-categories readme

  # Collect a Python project, keep Arcan output for inspection
  ./collect.sh --repo ~/projects/mllib --language PYTHON \\
               --output ~/data/mllib.jsonl \\
               --keep-arcan-output
EOF
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)               REPO_PATH="$2";          shift 2 ;;
        --language)           LANGUAGE="$2";            shift 2 ;;
        --output)             OUTPUT_FILE="$2";         shift 2 ;;
        --arcan-output)       ARCAN_OUTPUT_DIR="$2";   shift 2 ;;
        --keep-arcan-output)  KEEP_ARCAN_OUTPUT=1;     shift   ;;
        --exclude-categories) EXCLUDE_CATEGORIES="$2"; shift 2 ;;
        --max-bytes)          MAX_BYTES="$2";           shift 2 ;;
        --phase-a-depth)      PHASE_A_DEPTH="$2";      shift 2 ;;
        --phase-a-max)        PHASE_A_MAX="$2";         shift 2 ;;
        --phase-b-depth)      PHASE_B_DEPTH="$2";      shift 2 ;;
        --phase-b-max)        PHASE_B_MAX="$2";         shift 2 ;;
        --help|-h)            usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# ── Validate required args ─────────────────────────────────────────────────────
if [[ -z "$REPO_PATH" || -z "$LANGUAGE" || -z "$OUTPUT_FILE" ]]; then
    echo "Error: --repo, --language, and --output are required." >&2
    usage
    exit 1
fi

REPO_PATH="$(cd "$REPO_PATH" && pwd)"
REPO_NAME="$(basename "$REPO_PATH")"
REPO_PARENT="$(dirname "$REPO_PATH")"

OUTPUT_FILE="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$OUTPUT_FILE")"
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUTPUT_DIR"

if [[ -z "$ARCAN_OUTPUT_DIR" ]]; then
    ARCAN_OUTPUT_DIR="$OUTPUT_DIR"
fi
ARCAN_OUTPUT_DIR="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$ARCAN_OUTPUT_DIR")"
mkdir -p "$ARCAN_OUTPUT_DIR"

# ── Detect OS / Docker flags ──────────────────────────────────────────────────
DOCKER_PLATFORM_FLAG=""
OS_TYPE="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS_TYPE" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        echo "  Detected: macOS Apple Silicon (arm64) — adding --platform linux/amd64"
        DOCKER_PLATFORM_FLAG="--platform linux/amd64"
    else
        echo "  Detected: macOS Intel (x86_64)"
    fi
else
    echo "  Detected: Linux ($ARCH)"
fi

# ── Check Docker is available ──────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "Error: Docker is not installed or not in PATH." >&2
    echo "Install Docker Desktop from https://docs.docker.com/get-docker/" >&2
    exit 1
fi

# ── Step 1: Run Arcan ─────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 1/2: Running Arcan analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Repo:     $REPO_PATH"
echo "  Language: $LANGUAGE"
echo "  Output:   $ARCAN_OUTPUT_DIR/$REPO_NAME"
echo ""

# Arcan names its output after the directory passed via -i, so we mount
# the parent and use /data/repos/<name> as input path.
# Optional passthroughs: JAVA_TOOL_OPTIONS (e.g. "-Xmx5g") and ARCAN_DOCKER_ARGS
# (extra `docker run` flags, word-split, e.g. "--name arcan-foo --memory 7g").
DOCKER_EXTRA=()
[[ -n "${JAVA_TOOL_OPTIONS:-}" ]] && DOCKER_EXTRA+=(-e "JAVA_TOOL_OPTIONS=$JAVA_TOOL_OPTIONS")
[[ -n "${ARCAN_DOCKER_ARGS:-}" ]] && read -r -a _extra <<< "$ARCAN_DOCKER_ARGS" && DOCKER_EXTRA+=("${_extra[@]}")

docker run --rm --network none \
    $DOCKER_PLATFORM_FLAG \
    ${DOCKER_EXTRA[@]+"${DOCKER_EXTRA[@]}"} \
    -v "$REPO_PARENT:/data/repos:ro" \
    -v "$ARCAN_OUTPUT_DIR:/data/arcan_out" \
    "$ARCAN_IMAGE" \
    analyze \
      -i "/data/repos/$REPO_NAME" \
      -o /data/arcan_out \
      -l "$LANGUAGE" \
      --all \
      output.writeDependencyGraph=true \
      output.writeMembership=true \
      output.writeDependency=true

ARCAN_REPO_OUTPUT="$ARCAN_OUTPUT_DIR/arcanOutput/$REPO_NAME"
if [[ ! -d "$ARCAN_REPO_OUTPUT" ]]; then
    echo ""
    echo "Error: Arcan did not produce output at $ARCAN_REPO_OUTPUT" >&2
    echo "Check the Docker output above for errors." >&2
    exit 1
fi
echo ""
echo "  Arcan finished. Output at: $ARCAN_REPO_OUTPUT"

# ── Step 2: Collect data ──────────────────────────────────────────────────────
echo ""
echo "━━━ Step 2/2: Collecting data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Locate Python interpreter
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "Error: Python 3 is not installed or not in PATH." >&2
    echo "Install it from https://www.python.org/downloads/ or via 'brew install python3'." >&2
    exit 1
fi

PYTHON_VER="$($PYTHON --version 2>&1)"
echo "  Using: $PYTHON_VER"

# Build optional args
EXTRA_ARGS=()
if [[ -n "$EXCLUDE_CATEGORIES" ]]; then
    EXTRA_ARGS+=(--exclude-categories "$EXCLUDE_CATEGORIES")
fi

"$PYTHON" "$SCRIPT_DIR/collect_repo.py" \
    --repo-path "$REPO_PATH" \
    --arcan-output "$ARCAN_REPO_OUTPUT" \
    --language "$LANGUAGE" \
    --output "$OUTPUT_FILE" \
    --max-bytes "$MAX_BYTES" \
    --phaseA-depth "$PHASE_A_DEPTH" \
    --phaseA-max-files "$PHASE_A_MAX" \
    --phaseB-depth "$PHASE_B_DEPTH" \
    --phaseB-max-files "$PHASE_B_MAX" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ── Cleanup ───────────────────────────────────────────────────────────────────
if [[ "$KEEP_ARCAN_OUTPUT" -eq 0 ]]; then
    echo ""
    echo "  Removing Arcan output directory ($ARCAN_REPO_OUTPUT)..."
    rm -rf "$ARCAN_REPO_OUTPUT"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done!  Output: $OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
