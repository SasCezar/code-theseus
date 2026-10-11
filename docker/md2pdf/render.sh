#!/bin/sh
set -e

INPUT="$1"
OUTPUT="${2:-${INPUT%.md}.pdf}"

if [ -z "$INPUT" ]; then
    echo "Usage: render <input.md> [output.pdf]" >&2
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: file not found: $INPUT" >&2
    exit 1
fi

pandoc \
    --from=markdown \
    --pdf-engine=xelatex \
    --lua-filter=/usr/local/share/diagram.lua \
    --variable=geometry:margin=1in \
    --variable=colorlinks:true \
    --variable=linkcolor:blue \
    --output="$OUTPUT" \
    "$INPUT"
