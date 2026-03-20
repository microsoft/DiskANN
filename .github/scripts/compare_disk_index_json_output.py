#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
Compare two disk-index benchmark JSON files and emit a diff CSV.

This script takes baseline and branch (target) JSON files from the benchmark crate's
disk-index benchmarks and produces a CSV file comparing the metrics with deviation percentages.

The output format matches the CSV structure expected by benchmark_result_parse.py:
  Parent Span Name, Span Name, Stat Key, Stat Value (Target), Stat Value (Baseline), Deviation (%)

Usage:
    python compare_disk_index_json_output.py \\
        --baseline baseline/target/tmp/<dataset>_benchmark_crate_baseline.json \\
        --branch diskann_rust/target/tmp/<dataset>_benchmark_crate_target.json \\
        --out diskann_rust/target/tmp/<dataset>_change.csv
"""

import json
import csv
import argparse
from typing import List, Dict, Any, Optional


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return the parsed content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_deviation(baseline: float, target: float) -> str:
    """Calculate the percentage deviation from baseline to target."""
    try:
        if baseline != 0:
            dev = ((target - baseline) / baseline) * 100
            return f"{dev:.2f}"
        return ""
    except Exception:
        return ""


def extract_build_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract build metrics from the results structure."""
    if not results:
        return {}

    build = results.get("build", {})
    if not build:
        return {}

    metrics = {}

    # Total build time (in seconds)
    build_time = build.get("build_time")
    if build_time:
        # build_time is in microseconds, convert to seconds
        metrics["total_time"] = build_time / 1e6

    # Extract span metrics
    span_metrics = build.get("span_metrics", {})
    spans = span_metrics.get("spans", [])

    for span in spans:
        span_name = span.get("span_name", "")
        span_data = span.get("metrics", {})

        if span_name == "DiskIndexBuild-PqConstruction":
            metrics["pq_construction_time"] = span_data.get("duration_seconds", 0)
        elif span_name == "DiskIndexBuild-InmemIndexBuild":
            metrics["inmem_index_build_time"] = span_data.get("duration_seconds", 0)
        elif span_name == "DiskIndexBuild-DiskLayout":
            metrics["disk_layout_time"] = span_data.get("duration_seconds", 0)
        elif span_name == "disk-index-build":
            metrics["total_build_duration"] = span_data.get("duration_seconds", 0)

    return metrics


def extract_search_metrics(results: Dict[str, Any], search_l: int, beam_width: int) -> Dict[str, Any]:
    """Extract search metrics for a specific search_l value."""
    if not results:
        return {}

    search = results.get("search", {})
    if not search:
        return {}

    metrics = {}

    # Find the search result for the specified search_l
    search_results = search.get("search_results_per_l", [])
    for sr in search_results:
        if sr.get("search_l") == search_l:
            metrics["qps"] = sr.get("qps", 0)
            metrics["recall"] = sr.get("recall", 0)
            metrics["mean_latency"] = sr.get("mean_latency", 0)
            metrics["mean_ios"] = sr.get("mean_ios", 0)
            metrics["mean_comps"] = sr.get("mean_comparisons", 0)
            metrics["mean_hops"] = sr.get("mean_hops", 0)
            metrics["mean_io_time"] = sr.get("mean_io_time", 0)
            metrics["mean_cpus"] = sr.get("mean_cpu_time", 0)
            metrics["latency_95"] = sr.get("p999_latency", 0)  # Use p999 as proxy for 95th percentile
            break

    # Also try span metrics
    span_metrics = search.get("span_metrics", {})
    spans = span_metrics.get("spans", [])

    search_span_name = f"search-with-L={search_l}-bw={beam_width}"
    for span in spans:
        if span.get("span_name") == search_span_name:
            span_data = span.get("metrics", {})
            # Override with span metrics if they exist
            if "qps" in span_data:
                metrics["qps"] = span_data["qps"]
            if "recall" in span_data:
                metrics["recall"] = span_data["recall"]
            if "mean_latency" in span_data:
                metrics["mean_latency"] = span_data["mean_latency"]
            if "mean_ios" in span_data:
                metrics["mean_ios"] = span_data["mean_ios"]
            if "mean_comps" in span_data:
                metrics["mean_comps"] = span_data["mean_comps"]
            if "mean_hops" in span_data:
                metrics["mean_hops"] = span_data["mean_hops"]
            if "mean_io_time" in span_data:
                metrics["mean_io_time"] = span_data["mean_io_time"]
            if "mean_cpus" in span_data:
                metrics["mean_cpus"] = span_data["mean_cpus"]
            break

    return metrics


def make_rows(baseline_list: List[Dict], target_list: List[Dict]) -> List[List[str]]:
    """Generate comparison rows for the CSV output."""
    rows = []

    for baseline, target in zip(baseline_list, target_list):
        baseline_results = baseline.get("results", {})
        target_results = target.get("results", {})

        # Get input info for context
        inp = target.get("input", {})
        content = inp.get("content", {})
        search_phase = content.get("search_phase", {})

        # Determine search_l and beam_width for search metrics
        search_list = search_phase.get("search_list", [2000])
        beam_width = search_phase.get("beam_width", 4)

        # Use the first (or primary) search_l value
        primary_search_l = search_list[0] if search_list else 2000

        # Extract build metrics
        baseline_build = extract_build_metrics(baseline_results)
        target_build = extract_build_metrics(target_results)

        # Build metrics rows
        build_metrics = [
            ("total_time", "total build time (s)"),
            ("pq_construction_time", "PQ construction (s)"),
            ("inmem_index_build_time", "in-memory index build (s)"),
            ("disk_layout_time", "disk layout (s)"),
        ]

        for key, display_name in build_metrics:
            if key in target_build or key in baseline_build:
                target_val = target_build.get(key, 0)
                baseline_val = baseline_build.get(key, 0)
                rows.append([
                    "index-build statistics",
                    display_name,
                    key,
                    str(target_val),
                    str(baseline_val),
                    calc_deviation(baseline_val, target_val)
                ])

        # Extract search metrics for the primary search_l
        baseline_search = extract_search_metrics(baseline_results, primary_search_l, beam_width)
        target_search = extract_search_metrics(target_results, primary_search_l, beam_width)

        search_span_name = f"search-with-L={primary_search_l}-bw={beam_width}"

        # Search metrics rows
        search_metrics = [
            ("qps", "queries per second"),
            ("recall", "recall (%)"),
            ("mean_latency", "mean latency (μs)"),
            ("latency_95", "p999 latency (μs)"),
            ("mean_ios", "mean IOs"),
            ("mean_comps", "mean comparisons"),
            ("mean_hops", "mean hops"),
            ("mean_io_time", "mean IO time (μs)"),
            ("mean_cpus", "mean CPU time (μs)"),
        ]

        for key, display_name in search_metrics:
            if key in target_search or key in baseline_search:
                target_val = target_search.get(key, 0)
                baseline_val = baseline_search.get(key, 0)
                rows.append([
                    search_span_name,
                    display_name,
                    key,
                    str(target_val),
                    str(baseline_val),
                    calc_deviation(baseline_val, target_val)
                ])

    return rows


def write_csv(rows: List[List[str]], out_path: str):
    """Write the comparison rows to a CSV file."""
    header = [
        "Parent Span Name",
        "Span Name",
        "Stat Key",
        "Stat Value (Target)",
        "Stat Value (Baseline)",
        "Deviation (%)"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two disk-index benchmark JSONs and emit a diff CSV."
    )
    parser.add_argument("--baseline", "-b", required=True, help="Path to baseline JSON")
    parser.add_argument("--branch", "-r", required=True, help="Path to branch/target JSON")
    parser.add_argument("--out", "-o", required=True, help="Where to write output CSV")
    args = parser.parse_args()

    baseline_list = load_json(args.baseline)
    target_list = load_json(args.branch)

    if len(baseline_list) != len(target_list):
        raise ValueError(
            f"baseline/branch JSON arrays differ in length: {len(baseline_list)} vs {len(target_list)}"
        )

    rows = make_rows(baseline_list, target_list)
    write_csv(rows, args.out)
    print(f"✓ Written diff CSV to {args.out}")


if __name__ == "__main__":
    main()
