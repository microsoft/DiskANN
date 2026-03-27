#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
Benchmark Result Parser for GitHub Actions

Parses benchmark CSV results and validates against thresholds.
Posts comments to GitHub PRs when regressions are detected.

Usage:
    python benchmark_result_parse.py --mode pr --file results.csv
    python benchmark_result_parse.py --mode aa --file results.csv --data search

Environment Variables (for PR comments):
    GITHUB_TOKEN: GitHub token for API access
    GITHUB_REPOSITORY: Owner/repo (e.g., "microsoft/DiskANN")
    GITHUB_PR_NUMBER: Pull request number
    GITHUB_RUN_ID: Workflow run ID for linking to logs
"""

import csv
import os
import sys
import argparse
import json
from typing import Any

# Optional: requests for posting PR comments
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Data Structures
# =============================================================================

# Template for full benchmark data (build + search)
DATA_TEMPLATE_FULL = {
    "DiskIndexBuild-PqConstruction": {
        "duration_seconds": [],
        "peak_memory_usage": []
    },
    "DiskIndexBuild-InmemIndexBuild": {
        "duration_seconds": [],
        "peak_memory_usage": []
    },
    "search_disk_index-search_completed": {
        "duration_seconds": [],
        "peak_memory_usage": []
    },
    "disk_index_perf_test": {
        "total_duration_seconds": [],
    },
    "index-build statistics": {
        "total_time": [],
        "total_comparisons": [],
        "search_hops": []
    },
    "search-with-L=2000-bw=4": {
        "latency_95": [],
        "mean_latency": [],
        "mean_io_time": [],
        "mean_cpus": [],
        "qps": [],
        "mean_ios": [],
        "mean_comps": [],
        "mean_hops": [],
        "recall": []
    },
    "search-with-L=100-bw=4": {
        "latency_95": [],
        "mean_latency": [],
        "mean_io_time": [],
        "mean_cpus": [],
        "qps": [],
        "mean_ios": [],
        "mean_comps": [],
        "mean_hops": [],
        "recall": []
    },
    "search-with-L=200-bw=4": {
        "latency_95": [],
        "mean_latency": [],
        "mean_io_time": [],
        "mean_cpus": [],
        "qps": [],
        "mean_ios": [],
        "mean_comps": [],
        "mean_hops": [],
        "recall": []
    }
}
DATA_TEMPLATE_SEARCH = {
    "search_disk_index-search_completed": {
        "duration_seconds": [],
        "peak_memory_usage": []
    },
    "disk_index_perf_test": {
        "total_duration_seconds": [],
    },
    "search-with-L=2000-bw=4": {
        "latency_95": [],
        "mean_latency": [],
        "mean_io_time": [],
        "mean_cpus": [],
        "qps": [],
        "mean_ios": [],
        "mean_comps": [],
        "mean_hops": [],
        "recall": []
    },
    "search-with-L=100-bw=4": {
        "latency_95": [],
        "mean_latency": [],
        "mean_io_time": [],
        "mean_cpus": [],
        "qps": [],
        "mean_ios": [],
        "mean_comps": [],
        "mean_hops": [],
        "recall": []
    },
    "search-with-L=200-bw=4": {
        "latency_95": [],
        "mean_latency": [],
        "mean_io_time": [],
        "mean_cpus": [],
        "qps": [],
        "mean_ios": [],
        "mean_comps": [],
        "mean_hops": [],
        "recall": []
    }
}

# Thresholds for benchmark values
# Format: [threshold_percentage, direction, contract_value]
# - threshold_percentage: Maximum allowed deviation percentage
# - direction: 'GT' = higher is better, 'LT' = lower is better
# - contract_value: Promised performance value (empty string if none)
#
# For 'GT' metrics (like QPS, recall): regression if value decreases beyond threshold
# For 'LT' metrics (like latency, memory): regression if value increases beyond threshold
DATA_THRESHOLDS = {
    "DiskIndexBuild-PqConstruction": {
        "duration_seconds": [10, 'LT', ""],
        "peak_memory_usage": [10, 'LT', ""]
    },
    "DiskIndexBuild-InmemIndexBuild": {
        "duration_seconds": [10, 'LT', ""],
        "peak_memory_usage": [10, 'LT', ""]
    },
    "search_disk_index-search_completed": {
        "duration_seconds": [10, 'LT', ""],
        "peak_memory_usage": [10, 'LT', 1.42]
    },
    "disk_index_perf_test": {
        "total_duration_seconds": [10, 'LT', ""],
    },
    "index-build statistics": {
        # Calibrated from 5 GitHub runner runs (10 observations):
        #   Wikipedia: 35.9–37.2s, OpenAI: 23.0–76.4s (SQ_1_2.0 variance)
        # Contract: worst × 1.5 to absorb shared-runner variance
        "total_time": [10, 'LT', 115],
        "total_comparisons": [1, 'LT', ""],
        "search_hops": [1, 'LT', ""]
    },
    "search-with-L=2000-bw=4": {
        # Calibrated from 5 GitHub runner runs (10 observations):
        #   QPS: 9.56–9.75 (both datasets)
        #   Recall: wiki 99.87%, oai 99.67–99.91%
        #   mean_ios: ~2007 (deterministic)
        #   mean_comps: wiki ~27609, oai 21618–24733
        "latency_95": [10, 'LT', ""],
        "mean_latency": [10, 'LT', ""],
        "mean_io_time": [10, 'LT', ""],
        "mean_cpus": [15, 'LT', ""],  # wider threshold — CPU time is noisy on shared runners
        "qps": [10, 'GT', 6.5],
        "mean_ios": [1, 'LT', 2410],
        "mean_comps": [1, 'LT', 33200],
        "mean_hops": [1, 'LT', ""],
        "recall": [1, 'GT', 98.0]
    },
    "search-with-L=100-bw=4": {
        "latency_95": [10, 'LT', ""],
        "mean_latency": [10, 'LT', ""],
        "mean_io_time": [10, 'LT', ""],
        "mean_cpus": [15, 'LT', ""],  # wider threshold — CPU time is noisy on shared runners
        "qps": [10, 'GT', ""],
        "mean_ios": [10, 'LT', ""],
        "mean_comps": [10, 'LT', ""],
        "mean_hops": [10, 'LT', ""],
        "recall": [1, 'GT', ""]
    },
    "search-with-L=200-bw=4": {
        "latency_95": [10, 'LT', ""],
        "mean_latency": [10, 'LT', ""],
        "mean_io_time": [10, 'LT', ""],
        "mean_cpus": [15, 'LT', ""],  # wider threshold — CPU time is noisy on shared runners
        "qps": [10, 'GT', ""],
        "mean_ios": [10, 'LT', ""],
        "mean_comps": [10, 'LT', ""],
        "mean_hops": [10, 'LT', ""],
        "recall": [1, 'GT', ""]
    }
}


# =============================================================================
# CSV Parsing
# =============================================================================

def parse_csv(file_path: str, data: dict[str, dict[str, list]]) -> dict[str, dict[str, list]]:
    """
    Parse benchmark CSV file and populate data structure.

    CSV format produced by compare_disk_index_json_output.py:
        Column 0: Parent Span Name  (category, e.g. "index-build statistics")
        Column 1: Span Name         (display name, unused for matching)
        Column 2: Stat Key          (metric key, e.g. "qps")
        Column 3: Stat Value (Target)
        Column 4: Stat Value (Baseline)
        Column 5: Deviation (%)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row

        for row in reader:
            if len(row) < 6:
                continue

            category = row[0].strip()
            metric_name = row[2].strip()

            if category in data and metric_name in data[category]:
                # Append: [current_value, baseline_value, change_percentage]
                data[category][metric_name].append(row[3])  # target (current)
                data[category][metric_name].append(row[4])  # baseline
                data[category][metric_name].append(row[5])  # deviation %

    return data


def get_data_template(data_type: str) -> dict[str, dict[str, list]]:
    """Get a fresh copy of the data template."""
    import copy
    if data_type == 'search':
        return copy.deepcopy(DATA_TEMPLATE_SEARCH)
    return copy.deepcopy(DATA_TEMPLATE_FULL)


# =============================================================================
# Threshold Checking
# =============================================================================

def get_target_change_range(threshold: float, direction: str, mode: str) -> tuple[float, float]:
    """
    Calculate acceptable change range based on threshold and direction.
    
    Args:
        threshold: Maximum allowed deviation percentage
        direction: 'GT' (higher is better) or 'LT' (lower is better)
        mode: 'aa' (A/A test, symmetric) or 'pr' (PR test, directional)
    
    Returns:
        Tuple of (min_allowed, max_allowed) change percentages
    """
    if mode == 'aa':
        # A/A test: symmetric threshold
        return (-threshold, threshold)
    else:
        # PR test: directional threshold
        if direction == 'GT':
            # Higher is better: allow any improvement, flag regressions
            return (-threshold, float('inf'))
        else:
            # Lower is better: allow any improvement (negative change), flag increases
            return (float('-inf'), threshold)


def format_interval(start: float, end: float) -> str:
    """Format a numeric interval as a string."""
    start_str = '-inf' if start == float('-inf') else f"{start}%"
    end_str = 'inf' if end == float('inf') else f"{end}%"
    return f"({start_str} - {end_str})"


def is_change_threshold_failed(change: float, target_range: tuple[float, float]) -> bool:
    """Check if the change exceeds the allowed threshold range."""
    return change < target_range[0] or change > target_range[1]


def is_promise_broken(current_value: float, target_value: Any, direction: str) -> tuple[bool, str]:
    """
    Check if the current value violates a promised contract value.
    
    Returns:
        Tuple of (is_broken, formatted_target_value)
    """
    if target_value == "":
        return False, "N/A"
    
    target_value = float(target_value)
    
    if direction == 'GT':
        # Higher is better: current should be >= target
        if current_value < target_value:
            return True, f"> {target_value}"
    else:
        # Lower is better: current should be <= target
        if current_value > target_value:
            return True, f"< {target_value}"
    
    return False, str(target_value)


def get_outcome_message(threshold_failed: bool, promise_broken: bool) -> str:
    """Generate human-readable outcome message."""
    if threshold_failed and promise_broken:
        return 'Regression detected, Promise broken'
    elif promise_broken:
        return 'Promise broken'
    elif threshold_failed:
        return 'Regression detected'
    return 'OK'


def check_thresholds(
    data: dict[str, dict[str, list]],
    thresholds: dict[str, dict[str, list]],
    mode: str,
    run_id: str | None = None
) -> tuple[bool, str]:
    """
    Check all metrics against their thresholds.
    
    Returns:
        Tuple of (has_failures, failure_report_markdown)
    """
    failed_rows = []
    
    for category in data:
        for metric in data[category]:
            # Skip metrics without thresholds defined
            if category not in thresholds or metric not in thresholds[category]:
                print(f"Skipping {category}/{metric} - no threshold defined")
                continue
            
            values = data[category][metric]
            if not values:
                # No data for this metric in the CSV — skip silently
                continue
            
            # Parse values: [current, baseline, change%]
            try:
                value_current = float(values[0])
                value_baseline = float(values[1])
                change = float(values[2]) if values[2] else 0.0
            except (ValueError, IndexError) as e:
                print(f"ERROR: Failed to parse {category}/{metric}: {e}")
                return True, f"Parse error for {category}/{metric}"
            
            # Get threshold config
            threshold_config = thresholds[category][metric]
            threshold_pct = threshold_config[0]
            direction = threshold_config[1]
            contract_value = threshold_config[2]
            
            # Check thresholds
            target_range = get_target_change_range(threshold_pct, direction, mode)
            threshold_failed = is_change_threshold_failed(change, target_range)
            promise_broken, target_formatted = is_promise_broken(value_current, contract_value, direction)
            
            if threshold_failed:
                print(f"THRESHOLD FAILED: {category}/{metric} change={change}% allowed={format_interval(*target_range)}")
            if promise_broken:
                print(f"PROMISE BROKEN: {category}/{metric} value={value_current} required={target_formatted}")
            
            if threshold_failed or promise_broken:
                outcome = get_outcome_message(threshold_failed, promise_broken)
                failed_rows.append(
                    f"| {category}/{metric} | {value_baseline} | {value_current} | "
                    f"{target_formatted} | {change}% | {format_interval(*target_range)} | {outcome} |"
                )
    
    if failed_rows:
        # Build failure report
        logs_link = ""
        if run_id:
            repo = os.getenv('GITHUB_REPOSITORY', 'microsoft/DiskANN')
            logs_link = f"https://github.com/{repo}/actions/runs/{run_id}"
        
        report = "### ❌ Benchmark Check Failed\n\n"
        if logs_link:
            report += f"Please investigate the [workflow logs]({logs_link}) to determine if the failure is due to your changes.\n\n"
        
        report += "| Metric | Baseline | Current | Contract | Change | Allowed | Outcome |\n"
        report += "|--------|----------|---------|----------|--------|---------|--------|\n"
        report += "\n".join(failed_rows)
        
        return True, report
    
    return False, ""


# =============================================================================
# GitHub Integration
# =============================================================================

def post_github_pr_comment(comment: str) -> bool:
    """
    Post a comment to a GitHub pull request.
    
    Requires environment variables:
        GITHUB_TOKEN: Personal access token or GitHub Actions token
        GITHUB_REPOSITORY: Owner/repo format
        GITHUB_PR_NUMBER: Pull request number
    """
    if not HAS_REQUESTS:
        print("WARNING: 'requests' module not available, cannot post PR comment")
        return False
    
    token = os.getenv('GITHUB_TOKEN')
    repo = os.getenv('GITHUB_REPOSITORY')
    pr_number = os.getenv('GITHUB_PR_NUMBER')
    
    if not all([token, repo, pr_number]):
        print("WARNING: Missing GitHub environment variables for PR comment")
        print(f"  GITHUB_TOKEN: {'set' if token else 'missing'}")
        print(f"  GITHUB_REPOSITORY: {repo or 'missing'}")
        print(f"  GITHUB_PR_NUMBER: {pr_number or 'missing'}")
        return False
    
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    body = {"body": comment}
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        print(f"Successfully posted comment to PR #{pr_number}")
        return True
    except requests.RequestException as e:
        print(f"ERROR: Failed to post PR comment: {e}")
        return False


def write_github_step_summary(content: str) -> None:
    """Write content to GitHub Actions step summary."""
    summary_file = os.getenv('GITHUB_STEP_SUMMARY')
    if summary_file:
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(content)
            f.write("\n")


def write_github_output(name: str, value: str) -> None:
    """Write an output variable for GitHub Actions."""
    output_file = os.getenv('GITHUB_OUTPUT')
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"{name}={value}\n")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Parse benchmark results and validate against thresholds.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check PR benchmark results
    python benchmark_result_parse.py --mode pr --file results_change.csv
    
    # Check A/A test results (symmetric thresholds)
    python benchmark_result_parse.py --mode aa --file results_change.csv
    
    # Check search-only benchmarks
    python benchmark_result_parse.py --mode pr --file results_change.csv --data search
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='aa',
        choices=['aa', 'pr', 'lkg'],
        help='Benchmark mode: aa=A/A test (symmetric), pr=PR test (directional), lkg=last known good'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='both',
        choices=['both', 'search'],
        help='Type of benchmark data: both=full benchmark, search=search-only'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Path to CSV file (overrides FILE_PATH env var)'
    )
    parser.add_argument(
        '--no-comment',
        action='store_true',
        help='Skip posting PR comment even in pr mode'
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Get file path
    file_path = args.file or os.getenv('FILE_PATH')
    if not file_path:
        print("ERROR: No input file specified. Use --file or set FILE_PATH env var.")
        return 1
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return 1
    
    print(f"Benchmark mode: {args.mode}")
    print(f"Data type: {args.data}")
    print(f"Input file: {file_path}")
    
    # Parse CSV
    data_template = get_data_template(args.data)
    data = parse_csv(file_path, data_template)
    
    # Debug output
    print("\nParsed data:")
    print(json.dumps({k: {sk: sv for sk, sv in v.items() if sv} for k, v in data.items() if any(v.values())}, indent=2))
    
    # Check thresholds
    run_id = os.getenv('GITHUB_RUN_ID')
    has_failures, report = check_thresholds(data, DATA_THRESHOLDS, args.mode, run_id)
    
    if has_failures:
        print("\n" + report)
        
        # Write to GitHub step summary
        write_github_step_summary(report)
        
        # Post PR comment if in pr mode
        if args.mode == 'pr' and not args.no_comment:
            post_github_pr_comment(report)
        
        # Set output for downstream steps
        write_github_output('benchmark_failed', 'true')
        
        return 1
    
    print("\n✅ All benchmark values passed!")
    write_github_step_summary("### ✅ Benchmark Check Passed\n\nAll metrics within acceptable thresholds.")
    write_github_output('benchmark_failed', 'false')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
