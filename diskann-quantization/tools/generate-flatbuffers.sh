#!/bin/sh

set -eu

usage() {
    cat <<EOF
Usage: $(basename "$0") [FLATC]

Generate Rust FlatBuffers bindings in src/flatbuffers.

Arguments:
  FLATC  Path to flatc (default: search PATH)
EOF
}

if [ "$#" -gt 1 ]; then
    printf 'error: expected at most one argument\n\n' >&2
    usage >&2
    exit 2
fi

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    "")
        flatc=flatc
        ;;
    *)
        flatc=$1
        ;;
esac

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output="$root/src/flatbuffers"

printf 'Generating FlatBuffers bindings with %s\n' "$flatc"
printf 'Output directory: %s\n' "$output"

"$flatc" \
    --rust \
    --rust-module-root-file \
    -I "$root/schemas" \
    -o "$output" \
    "$root"/schemas/*.fbs

rm "$output/mod.rs"

printf 'Adding license headers to generated files\n'
generated_count=0
for generated in "$output"/*/*_generated.rs; do
    {
        cat <<'EOF'
/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

EOF
        cat "$generated"
    } >"$generated.with-license"
    mv "$generated.with-license" "$generated"
    generated_count=$((generated_count + 1))
done

printf 'Generated %d FlatBuffers binding files\n' "$generated_count"
