#!/bin/sh
# Called once per file by render_batch.sh via xargs -P.
# Writes one status line to $STATS_DIR and prints progress to stdout.
#
# Usage: render_one <input_dir> <output_dir> <stats_dir> <mermaid_cache> <src_file>

INPUT_DIR="$1"
OUTPUT_DIR="$2"
STATS_DIR="$3"
MERMAID_CACHE="$4"
src="$5"

relpath="${src#${INPUT_DIR}/}"
out="$OUTPUT_DIR/${relpath%.md}.pdf"
mkdir -p "$(dirname "$out")"

tmp=$(mktemp)
trap 'rm -f "$tmp" "${tmp}.u" "${tmp}.m" "${tmp}.stderr"' EXIT
t_start=$(date +%s)

# ── Strip LLM thinking block ──────────────────────────────────────────────────
if grep -q '</think>' "$src" 2>/dev/null; then
    awk '/^<\/think>/{found=1; next} found{print}' "$src" > "$tmp"
elif grep -q 'assistantfinal' "$src" 2>/dev/null; then
    awk '/assistantfinal/{sub(/.*assistantfinal/, ""); print; found=1; next} found{print}' "$src" > "$tmp"
else
    cp "$src" "$tmp"
fi

# ── Unwrap outer ```markdown...``` fence (some models wrap entire output) ─────
if grep -q '^```markdown$' "$tmp" 2>/dev/null; then
    awk '
        /^```markdown$/ && !unwrapped { unwrapped=1; next }
        { lines[++n]=$0 }
        END {
            if (unwrapped && lines[n]=="```") n--
            for (i=1; i<=n; i++) print lines[i]
        }
    ' "$tmp" > "${tmp}.u" && mv "${tmp}.u" "$tmp"
fi

# ── Substitute pre-rendered mermaid PNGs (skips lua filter / Chromium) ────────
file_key=$(printf '%s' "$relpath" | sed 's|/|__|g; s|\.md$||')
awk -v cache="$MERMAID_CACHE" -v fkey="$file_key" '
    /^```mermaid$/ { in_m=1; n++; next }
    in_m && /^```$/ {
        in_m=0
        png = cache "/" fkey "/" n ".pdf"
        if (system("test -f \"" png "\"") == 0)
            print "![](" png ")"
        # else: silently drop failed diagrams rather than embed broken fences
        next
    }
    in_m { next }
    { print }
' "$tmp" > "${tmp}.m" && mv "${tmp}.m" "$tmp"

# ── Render helper (no lua filter — mermaid already substituted above) ─────────
_pandoc() {
    # yaml_metadata_block is disabled: our docs use bare "---" as a markdown
    # horizontal-rule section divider (not just at the top of the file), and
    # pandoc's default markdown reader pairs up ANY two "---"/"..." lines as
    # a YAML metadata block wherever they occur -- silently swallowing
    # everything between them and failing to parse it as YAML.
    pandoc \
        --from=markdown-yaml_metadata_block \
        --pdf-engine=xelatex \
        --pdf-engine-opt=-interaction=batchmode \
        --pdf-engine-opt=-halt-on-error \
        --variable=geometry:margin=1in \
        --variable=colorlinks:true \
        --variable=linkcolor:blue \
        --output="$out" \
        "$1" 2>"${tmp}.stderr"
}

# ── Mermaid fix (kept for the retry pass on other pandoc/latex issues) ────────
fix_mermaid() {
    local f="$1"
    sed -i 's/<br[[:space:]]*\/*>/\\n/g' "$f"
    sed -i 's/<[^>]*>//g' "$f"
}

# ── Attempt 1 ─────────────────────────────────────────────────────────────────
status="fail"
if _pandoc "$tmp"; then
    status="ok"
else
    fix_mermaid "$tmp"
    if _pandoc "$tmp"; then
        status="fixed"
    else
        cp "${tmp}.stderr" "$out.err" 2>/dev/null || true
    fi
fi

# ── Report ────────────────────────────────────────────────────────────────────
elapsed=$(( $(date +%s) - t_start ))
safe_name=$(printf '%s' "$relpath" | tr '/' '_')
printf '%s\t%s\t%d\n' "$relpath" "$status" "$elapsed" > "$STATS_DIR/${safe_name}.status"
printf '[%-5s %3ds] %s\n' "$status" "$elapsed" "$relpath"
