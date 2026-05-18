#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Compare two cargo-llvm-lines output files and produce a markdown regression report.
#
# Usage:
#   ./compare-llvm-lines.sh <baseline.txt> <current.txt>

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <baseline.txt> <current.txt>" >&2
  exit 1
fi

baseline_file="$1"
current_file="$2"

baseline_total=$(awk '/\(TOTAL\)/{print $1}' "$baseline_file")
current_total=$(awk '/\(TOTAL\)/{print $1}' "$current_file")

if [ "$baseline_total" -eq 0 ]; then
  echo "Error: baseline total is 0, cannot compute growth." >&2
  exit 1
fi

delta_total=$(( current_total - baseline_total ))

# Parse llvm-lines output into "lines\tfunction_name" format.
# Skip header (3 lines) and TOTAL row. Extract the first number (lines count)
# and the function name (everything after the second parenthesized group).
parse() {
  awk 'NR>3 && !/TOTAL/{
    lines = $1
    sub(/^[^)]*\)[^)]*\) */, "")
    print lines "\t" $0
  }' "$1" | sort -t$'\t' -k2
}

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
baseline_parsed_file="$tmpdir/llvm_baseline_parsed.txt"
current_parsed_file="$tmpdir/llvm_current_parsed.txt"

parse "$baseline_file" > "$baseline_parsed_file"
parse "$current_file" > "$current_parsed_file"

echo "| Delta | Current | Baseline | Function |"
echo "|-------|---------|----------|----------|"
printf "| %+d | %s | %s | (TOTAL) |\n" "$delta_total" "$current_total" "$baseline_total"
join -t$'\t' -j 2 -a1 -e 0 -o 1.2,1.1,2.1 \
  "$current_parsed_file" "$baseline_parsed_file" | \
  awk -F'\t' '{delta=$2-$3; printf "%d\t%s\t%s\t%s\n", delta, $2, $3, $1}' | \
  sort -t$'\t' -rn | \
  awk -F'\t' '{printf "| %+d | %s | %s | %s |\n", $1, $2, $3, $4}'
echo ""
