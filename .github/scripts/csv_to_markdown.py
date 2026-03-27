#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Convert a CSV file to a Markdown table and optionally append to GitHub Step Summary."""

import argparse
import csv
import os
import sys


def csv_to_markdown(csv_path: str) -> str:
    """Convert a CSV file to a Markdown table string."""
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    return "\n".join(" | ".join(r) for r in [header, sep] + rows[1:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Input CSV file path")
    parser.add_argument("--md", required=True, help="Output Markdown file path")
    parser.add_argument("--title", default="", help="Section title for GitHub Step Summary")
    args = parser.parse_args()

    md = csv_to_markdown(args.csv)
    if not md:
        print("No data")
        return 0

    with open(args.md, "w") as f:
        f.write(md + "\n")

    # Append to GitHub Step Summary if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path and args.title:
        with open(summary_path, "a") as f:
            f.write(f"### {args.title}\n")
            f.write(md + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
