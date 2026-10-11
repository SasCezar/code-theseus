#!/usr/bin/env bash
# Run Arcan analysis (Docker) over every repo listed in a language TSV
# (produced by scripts/python/filter_dataset.py). Writes to data/raw/arcanOutput/.
#
# Usage:
#   ./scripts/bash/run_arcan_docker.sh repos.tsv
#
# Unthrottled by default (one container per TSV row, up to JOBS at a time).
# Set BUDGET_GB to enable the memory-aware throttle (token-bucket + a
# MemAvailable gate + per-container --memory caps) for large batches on a
# shared/limited-RAM host — this replaces the old run_arcan_docker_mem.sh,
# which was the same script with that throttle always on.
set -euo pipefail

TSV="${1:-}"
[[ -n "$TSV" && -f "$TSV" ]] || { echo "Usage: $0 repos.tsv" >&2; exit 2; }

# Feeder parallelism (xargs). With BUDGET_GB set, effective concurrency is
# further bounded by the token + MemAvailable gate below.
JOBS="${JOBS:-$(nproc)}"
TIMEOUT="${TIMEOUT:-3h}"
KILL_AFTER="${KILL_AFTER:-2m}"

IMAGE="${IMAGE:-ghcr.io/arcan-tech/arcan-2-cli-trial:latest}"
JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xmx24g -XX:MaxRAMPercentage=80}"

# -----------------------------
# Memory/throughput controls (opt-in: set BUDGET_GB to enable)
# -----------------------------
# Total "budget" for concurrently running analyses (leave headroom for OS/page
# cache). For a 198GB host, start around 140-160. Unset/0 = throttle disabled.
BUDGET_GB="${BUDGET_GB:-0}"

# Each analysis consumes this many "tokens" (GB). Effective concurrency ~= BUDGET_GB / TOKEN_GB.
TOKEN_GB="${TOKEN_GB:-8}"

# Do not start a new container unless MemAvailable >= this many GB.
MIN_AVAIL_GB="${MIN_AVAIL_GB:-35}"

# Per-container hard cap (localizes failures). Keep swap equal to memory to avoid swap thrash.
CONTAINER_MEM="${CONTAINER_MEM:-24g}"
CONTAINER_MEMSWAP="${CONTAINER_MEMSWAP:-24g}"

# Optional: prevent runaway process counts inside container (0 disables).
PIDS_LIMIT="${PIDS_LIMIT:-0}"

DATA_MOUNT="${DATA_MOUNT:-$(pwd)}"
IN_BASE="${IN_BASE:-/data/data/repository}"
OUT_BASE="${OUT_BASE:-/data/data/raw}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-./logs_arcan_${RUN_ID}}"
RESULTS_CSV="${RESULTS_CSV:-./results_arcan_${RUN_ID}.csv}"

SKIP_FIRST="${SKIP_FIRST:-0}"

THROTTLED=false
[[ "$BUDGET_GB" -gt 0 ]] && THROTTLED=true

mkdir -p "$LOG_DIR"
echo "run_id,ts_utc,url,name,language,exit_code,duration_seconds,timeout,log_file" > "$RESULTS_CSV"

echo "[INFO] Pulling image once: $IMAGE"
docker pull "$IMAGE" >/dev/null

if [[ "$THROTTLED" == "true" ]]; then
  docker container prune -f >/dev/null 2>&1 || true

  TOKENS=$(( BUDGET_GB / TOKEN_GB ))
  (( TOKENS >= 1 )) || { echo "[ERROR] BUDGET_GB ($BUDGET_GB) too small for TOKEN_GB ($TOKEN_GB)" >&2; exit 2; }

  SEMDIR="${SEMDIR:-/tmp/arcan_sema_${RUN_ID}}"
  mkdir -p "$SEMDIR"
  if ! ls "$SEMDIR"/* >/dev/null 2>&1; then
    for i in $(seq 1 "$TOKENS"); do : > "$SEMDIR/$i"; done
  fi

  acquire_token() {
    while :; do
      for f in "$SEMDIR"/*; do
        # fd 9 will be held open+locked for the duration of the run_one() call
        if exec 9>"$f" && flock -n 9; then
          echo "$f"
          return 0
        fi
      done
      sleep 1
    done
  }

  release_token() {
    flock -u 9 >/dev/null 2>&1 || true
    exec 9>&- >/dev/null 2>&1 || true
  }

  wait_for_ram() {
    local min_avail_gb="$MIN_AVAIL_GB"
    while :; do
      local avail_kb avail_gb
      avail_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
      avail_gb=$(( avail_kb / 1024 / 1024 ))
      (( avail_gb >= min_avail_gb )) && return 0
      sleep 2
    done
  }
else
  TOKENS=0
  acquire_token() { :; }
  release_token() { :; }
  wait_for_ram() { :; }
fi

extract_name() {
  local url="$1"
  local p="${url#https://github.com/}"
  local user="${p%%/*}"
  local repo="${p#*/}"
  repo="${repo%%/*}"
  [[ -n "$user" && -n "$repo" ]] || return 1
  printf '%s|%s' "$user" "$repo"
}

run_one() {
  local url="$1"
  local lang="${2:-}"

  local name
  if ! name="$(extract_name "$url")"; then
    local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s,%s,%q,%q,%q,%s,%s,%s,%q\n' \
      "$RUN_ID" "$ts" "$url" "" "$lang" "2" "0" "false" "" >> "$RESULTS_CSV"
    printf '[SKIP] %s url=%s (invalid github url)\n' "$ts" "$url" >&2
    return 0
  fi

  local in_dir="${IN_BASE}/${name}/"
  local out_dir="${OUT_BASE}"
  local safe="${name//|/__}"
  local log_file="${LOG_DIR}/${safe}.log"
  local ctag="arcan_${safe}_$$_$(date +%s%N)"

  # Skip analysis if output directory already exists
  local existing_out="${DATA_MOUNT}/data/raw/arcanOutput/${name}"
  if [[ -d "$existing_out" ]]; then
    local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s,%s,%q,%q,%q,%s,%s,%s,%q\n' \
      "$RUN_ID" "$ts" "$url" "$name" "$lang" "0" "0" "false" "SKIPPED_EXISTS" >> "$RESULTS_CSV"
    printf '[SKIP] ts_utc=%s name=%s (output exists: %s)\n' \
      "$ts" "$name" "$existing_out" >&2
    return 0
  fi

  local token_file=""
  if [[ "$THROTTLED" == "true" ]]; then
    token_file="$(acquire_token)"
    trap 'release_token' RETURN
    wait_for_ram
  fi

  local start end dur exit_code timeout_hit="false"
  start="$(date +%s)"

  printf '[START] ts_utc=%s name=%s lang=%s log=%s%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "${lang:-}" "$log_file" \
    "${token_file:+ (token=$(basename "$token_file"))}" >&2

  {
    echo "[INFO] url=$url"
    echo "[INFO] name=$name"
    echo "[INFO] lang=$lang"
    echo "[INFO] in_dir=$in_dir"
    echo "[INFO] out_dir=$out_dir"
    echo "[INFO] container=$ctag"
    echo "[INFO] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[INFO] throttled=$THROTTLED"
    echo
  } > "$log_file"

  local -a DOCKER_ARGS
  DOCKER_ARGS=(docker run --rm --name "$ctag" -e "JAVA_TOOL_OPTIONS=$JAVA_TOOL_OPTIONS" -v "${DATA_MOUNT}:/data")
  if [[ "$THROTTLED" == "true" ]]; then
    DOCKER_ARGS+=(--memory="$CONTAINER_MEM" --memory-swap="$CONTAINER_MEMSWAP")
    [[ "$PIDS_LIMIT" != "0" ]] && DOCKER_ARGS+=(--pids-limit="$PIDS_LIMIT")
  fi

  set +e
  timeout --preserve-status --kill-after="$KILL_AFTER" "$TIMEOUT" \
    "${DOCKER_ARGS[@]}" \
      "$IMAGE" \
      analyze -i "$in_dir" -o "$out_dir" ${lang:+-l "$lang"} --all \
      --remote "$url" \
      output.writeDependencyGraph=true output.writeMembership=true output.writeDependency=true \
    >> "$log_file" 2>&1
  exit_code=$?
  set -e

  if [[ "$exit_code" -eq 124 || "$exit_code" -eq 137 ]]; then
    timeout_hit="true"
    docker rm -f "$ctag" >/dev/null 2>&1 || true
  elif [[ "$THROTTLED" == "true" ]]; then
    docker rm -f "$ctag" >/dev/null 2>&1 || true
  fi

  end="$(date +%s)"
  dur=$(( end - start ))

  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s,%s,%q,%q,%q,%s,%s,%s,%q\n' \
    "$RUN_ID" "$ts" "$url" "$name" "$lang" "$exit_code" "$dur" "$timeout_hit" "$log_file" >> "$RESULTS_CSV"

  printf '[END]   ts_utc=%s name=%s exit=%s dur_s=%s timeout=%s\n' \
    "$ts" "$name" "$exit_code" "$dur" "$timeout_hit" >&2
}

export -f extract_name run_one acquire_token release_token wait_for_ram
export RUN_ID TIMEOUT KILL_AFTER IMAGE JAVA_TOOL_OPTIONS DATA_MOUNT IN_BASE OUT_BASE LOG_DIR RESULTS_CSV
export THROTTLED CONTAINER_MEM CONTAINER_MEMSWAP PIDS_LIMIT SEMDIR

echo "[INFO] Starting: jobs=$JOBS timeout=$TIMEOUT kill_after=$KILL_AFTER skip_first=$SKIP_FIRST throttled=$THROTTLED"
[[ "$THROTTLED" == "true" ]] && echo "[INFO] Throttle: budget_gb=$BUDGET_GB token_gb=$TOKEN_GB tokens=$TOKENS min_avail_gb=$MIN_AVAIL_GB mem=$CONTAINER_MEM"

awk -F'\t' 'NF>=1 && $1!="" && $1!~/^[[:space:]]*#/ { print $1 "\t" $2 }' "$TSV" \
  | {
      if [[ "$SKIP_FIRST" -gt 0 ]]; then
        tail -n +"$((SKIP_FIRST + 1))"
      else
        cat
      fi
    } \
  | xargs -r -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"' || true

echo "[INFO] Done."
echo "[INFO] Logs:    $LOG_DIR"
echo "[INFO] Results: $RESULTS_CSV"
