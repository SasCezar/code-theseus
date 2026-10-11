#!/usr/bin/env bash
set -euo pipefail

TSV="${1:-}"
[[ -n "$TSV" && -f "$TSV" ]] || { echo "Usage: $0 repos.tsv" >&2; exit 2; }

DATA_MOUNT="${DATA_MOUNT:-$(pwd)}"
DEST_BASE="${DEST_BASE:-${DATA_MOUNT}/data/data/repository}"

JOBS="${JOBS:-$(nproc)}"
DEPTH="${DEPTH:-1}"

TIMEOUT="${TIMEOUT:-3h}"
KILL_AFTER="${KILL_AFTER:-2m}"

GITHUB_TOKEN="${GITHUB_TOKEN:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-./logs_prefetch_${RUN_ID}}"
RESULTS_CSV="${RESULTS_CSV:-./results_prefetch_${RUN_ID}.csv}"

mkdir -p "$DEST_BASE" "$LOG_DIR"
echo "run_id,ts_utc,url,name,exit_code,duration_seconds,timeout,log_file,action" > "$RESULTS_CSV"

extract_name() {
  local url="$1"
  local p="${url#https://github.com/}"
  local user="${p%%/*}"
  local repo="${p#*/}"
  repo="${repo%%/*}"
  [[ -n "$user" && -n "$repo" ]] || return 1
  printf '%s|%s' "$user" "$repo"
}

git_url_with_token() {
  local url="$1"
  if [[ -n "$GITHUB_TOKEN" ]]; then
    echo "${url/https:\/\/github.com/https:\/\/${GITHUB_TOKEN}@github.com}"
  else
    echo "$url"
  fi
}

clone_one() {
  local url="$1"
  local _lang="${2:-}"

  local ts name safe repo_dir log_file start end dur exit_code timeout_hit="false" action

  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if ! name="$(extract_name "$url")"; then
    printf '%s,%s,%q,%q,%s,%s,%s,%q,%q\n' \
      "$RUN_ID" "$ts" "$url" "" "2" "0" "false" "" "invalid_url" >> "$RESULTS_CSV"
    printf '[SKIP] %s url=%s (invalid github url)\n' "$ts" "$url" >&2
    return 0
  fi

  safe="${name//|/__}"
  repo_dir="${DEST_BASE}/${name}"
  log_file="${LOG_DIR}/${safe}.log"

  start="$(date +%s)"

  printf '[START] ts_utc=%s name=%s log=%s\n' "$ts" "$name" "$log_file" >&2
  {
    echo "[INFO] url=$url"
    echo "[INFO] name=$name"
    echo "[INFO] repo_dir=$repo_dir"
    echo "[INFO] depth=$DEPTH"
    echo "[INFO] start_utc=$ts"
    echo
  } > "$log_file"

  mkdir -p "$(dirname "$repo_dir")"

  if [[ -d "$repo_dir/.git" ]]; then
    action="skip_exists"
    exit_code=0
  else
    action="clone"
    local gurl
    gurl="$(git_url_with_token "$url")"

    set +e
    timeout --preserve-status --kill-after="$KILL_AFTER" "$TIMEOUT" \
      git clone --no-tags --depth "$DEPTH" "$gurl" "$repo_dir" >>"$log_file" 2>&1
    exit_code=$?
    set -e

    if [[ "$exit_code" -eq 124 || "$exit_code" -eq 137 ]]; then
      timeout_hit="true"
      # best-effort cleanup of partial clone
      rm -rf "$repo_dir" >/dev/null 2>&1 || true
    elif [[ "$exit_code" -ne 0 ]]; then
      # cleanup failed clone
      rm -rf "$repo_dir" >/dev/null 2>&1 || true
    fi
  fi

  end="$(date +%s)"
  dur=$(( end - start ))

  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s,%s,%q,%q,%s,%s,%s,%q,%q\n' \
    "$RUN_ID" "$ts" "$url" "$name" "$exit_code" "$dur" "$timeout_hit" "$log_file" "$action" >> "$RESULTS_CSV"

  if [[ "$exit_code" -eq 0 ]]; then
    printf '[END]   ts_utc=%s name=%s action=%s dur_s=%s\n' "$ts" "$name" "$action" "$dur" >&2
  else
    printf '[END]   ts_utc=%s name=%s action=%s exit=%s dur_s=%s timeout=%s log=%s\n' \
      "$ts" "$name" "$action" "$exit_code" "$dur" "$timeout_hit" "$log_file" >&2
  fi
}

export -f extract_name git_url_with_token clone_one
export RUN_ID TIMEOUT KILL_AFTER DATA_MOUNT DEST_BASE DEPTH GITHUB_TOKEN LOG_DIR RESULTS_CSV

echo "[INFO] Prefetch(clone-only) starting: jobs=$JOBS depth=$DEPTH dest=$DEST_BASE timeout=$TIMEOUT kill_after=$KILL_AFTER"

awk -F'\t' 'NF>=1 && $1!="" && $1!~/^[[:space:]]*#/ { print $1 "\t" $2 }' "$TSV" \
  | xargs -r -P "$JOBS" -n 2 bash -c 'clone_one "$0" "$1"' || true

echo "[INFO] Done."
echo "[INFO] Logs:    $LOG_DIR"
echo "[INFO] Results: $RESULTS_CSV"
