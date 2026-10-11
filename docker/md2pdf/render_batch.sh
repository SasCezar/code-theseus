#!/bin/sh
# Batch entrypoint — runs inside the container.
# Processes all .md files in INPUT_DIR in parallel, writes PDFs to OUTPUT_DIR.
#
# Usage: render_batch <input_dir> <output_dir>
# Env:   JOBS  number of parallel renders (default: 4)

INPUT_DIR="${1:-/docs}"
OUTPUT_DIR="${2:-/output}"
JOBS="${JOBS:-4}"

mkdir -p "$OUTPUT_DIR"
STATS_DIR=$(mktemp -d)
MERMAID_CACHE=$(mktemp -d)

total=$(find "$INPUT_DIR" -name '*.md' | wc -l | tr -d ' ')
printf 'Rendering %d files with %d parallel jobs...\n\n' "$total" "$JOBS"

printf '── Pre-rendering mermaid diagrams (single Chromium session) ──\n'
node /usr/local/bin/batch_mermaid.js "$INPUT_DIR" "$MERMAID_CACHE"
printf '\n'

t_wall=$(date +%s)
find "$INPUT_DIR" -name '*.md' -print0 \
    | xargs -0 -P "$JOBS" -n 1 \
        /usr/local/bin/render_one "$INPUT_DIR" "$OUTPUT_DIR" "$STATS_DIR" "$MERMAID_CACHE"
t_elapsed=$(( $(date +%s) - t_wall ))

# ── Collect stats ─────────────────────────────────────────────────────────────
TSV="$OUTPUT_DIR/render_stats.tsv"
printf 'file\tstatus\n' > "$TSV"
cat "$STATS_DIR"/*.status >> "$TSV" 2>/dev/null || true
rm -rf "$STATS_DIR"

n_ok=$(    awk -F'\t' '$2=="ok"'    "$TSV" | wc -l | tr -d ' ')
n_fixed=$( awk -F'\t' '$2=="fixed"' "$TSV" | wc -l | tr -d ' ')
n_fail=$(  awk -F'\t' '$2=="fail"'  "$TSV" | wc -l | tr -d ' ')

pct() { awk "BEGIN{printf \"%.1f\", ($1)*100/($2)}"; }

printf '\n'
printf '%-10s %4d / %-4d  (%s%%)\n' ok     "$n_ok"    "$total" "$(pct "$n_ok"    "$total")"
printf '%-10s %4d / %-4d  (%s%%)\n' fixed  "$n_fixed" "$total" "$(pct "$n_fixed" "$total")"
printf '%-10s %4d / %-4d  (%s%%)\n' failed "$n_fail"  "$total" "$(pct "$n_fail"  "$total")"

if [ "$n_fail" -gt 0 ]; then
    printf '\nFailures by directory:\n'
    awk -F'\t' '$2=="fail" && $1~/\// {
        split($1, a, "/"); dir=a[1]
        fail[dir]++; total_dir[dir]++
    }
    $2!="fail" && $1~/\// {
        split($1, a, "/"); dir=a[1]
        total_dir[dir]++
    }
    END {
        for (d in fail)
            printf "  %-35s  %4d / %-4d  (%.1f%%)\n", d, fail[d], total_dir[d], fail[d]*100/total_dir[d]
    }' "$TSV" | sort
fi

avg_s=$(awk -F'\t' 'NR>1{sum+=$3; n++} END{if(n>0) printf "%.1f", sum/n; else print "?"}' "$TSV")
rate=$(awk "BEGIN{if($t_elapsed>0) printf \"%.2f\", ($n_ok+$n_fixed+$n_fail)/$t_elapsed; else print \"?\"}")
printf '\nWall time: %ds  |  avg per file: %ss  |  throughput: %s docs/s\n' \
    "$t_elapsed" "$avg_s" "$rate"
printf 'Stats: /output/render_stats.tsv\n'
