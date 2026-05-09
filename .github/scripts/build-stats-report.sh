#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Collect build-stats artifacts from recent CI runs and generate an HTML report.
#
# Usage: build-stats-report.sh [github_repository] [collected_dir] [report_dir]
#
# Examples:
#   build-stats-report.sh
#   build-stats-report.sh microsoft/DiskANN collected report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GITHUB_REPOSITORY="${1:-microsoft/DiskANN}"

if [ -z "${2:-}" ]; then
  WORK_DIR=$(mktemp -d)
  COLLECTED_DIR="$WORK_DIR/collected"
  REPORT_DIR="$WORK_DIR/report"
  mkdir -p "$COLLECTED_DIR" "$REPORT_DIR"
else
  COLLECTED_DIR="$2"
  REPORT_DIR="${3:?report_dir is required when collected_dir is provided}"
fi

SINCE=$(date -u -d '30 days ago' '+%Y-%m-%dT%H:%M:%SZ')

gh api --paginate \
  "repos/$GITHUB_REPOSITORY/actions/workflows/build-stats.yml/runs?status=success&created=>=$SINCE&per_page=100" \
  --jq '.workflow_runs[] | [.id, .created_at, .head_sha] | @tsv' \
  > runs.tsv || true

if [ ! -s runs.tsv ]; then
  echo "::warning::No successful build-stats runs found in the last 30 days"
  exit 1
fi

echo "Found $(wc -l < runs.tsv) runs"
cp runs.tsv "$COLLECTED_DIR/runs.tsv"

while IFS=$'\t' read -r run_id created_at head_sha; do
  gh run download "$run_id" --repo "$GITHUB_REPOSITORY" \
    --name build-stats --dir "$COLLECTED_DIR/$run_id" 2>/dev/null \
    || echo "::warning::Skipping run $run_id (artifact expired)"
done < runs.tsv

python3 "$SCRIPT_DIR/build-stats-report-data.py" "$COLLECTED_DIR" "$REPORT_DIR"
cp "$SCRIPT_DIR/../reports/build-stats-report.html" "$REPORT_DIR/"

echo ""
echo "Report: $REPORT_DIR/build-stats-report.html"
