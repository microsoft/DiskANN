#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
Benchmark Validator for GitHub Actions

Compares two benchmark JSON outputs (baseline vs target), checks thresholds,
writes a Markdown summary, and optionally posts a PR comment on failure.

This single script replaces the previous three-step pipeline:
  compare_disk_index_json_output.py → csv_to_markdown.py → benchmark_result_parse.py

Usage:
    # PR mode (directional thresholds, posts PR comment on failure)
    python benchmark_validate.py --mode pr --baseline baseline.json --target target.json

    # A/A mode (symmetric thresholds)
    python benchmark_validate.py --mode aa --baseline baseline.json --target target.json

Environment Variables (for PR comments):
    GITHUB_TOKEN: GitHub token for API access
    GITHUB_REPOSITORY: Owner/repo (e.g., "microsoft/DiskANN")
    GITHUB_PR_NUMBER: Pull request number
    GITHUB_RUN_ID: Workflow run ID for linking to logs
    GITHUB_STEP_SUMMARY: Path to step summary file
"""

import json
import os
import sys
import argparse
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError


# =============================================================================
# JSON Extraction
# =============================================================================

def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_build_metrics(results: dict) -> dict[str, float]:
    build = results.get("build", {})
    if not build:
        return {}

    metrics: dict[str, float] = {}

    build_time = build.get("build_time")
    if build_time:
        metrics["total_time"] = build_time / 1e6  # μs → s

    for span in build.get("span_metrics", {}).get("spans", []):
        name = span.get("span_name", "")
        data = span.get("metrics", {})
        if name == "DiskIndexBuild-PqConstruction":
            metrics["pq_construction_time"] = data.get("duration_seconds", 0)
        elif name == "DiskIndexBuild-InmemIndexBuild":
            metrics["inmem_index_build_time"] = data.get("duration_seconds", 0)
        elif name == "DiskIndexBuild-DiskLayout":
            metrics["disk_layout_time"] = data.get("duration_seconds", 0)

    return metrics


def extract_search_metrics(results: dict, search_l: int, beam_width: int) -> dict[str, float]:
    search = results.get("search", {})
    if not search:
        return {}

    metrics: dict[str, float] = {}

    # From search_results_per_l
    for sr in search.get("search_results_per_l", []):
        if sr.get("search_l") == search_l:
            metrics["qps"] = sr.get("qps", 0)
            metrics["recall"] = sr.get("recall", 0)
            metrics["mean_latency"] = sr.get("mean_latency", 0)
            metrics["mean_ios"] = sr.get("mean_ios", 0)
            metrics["mean_comps"] = sr.get("mean_comparisons", 0)
            metrics["mean_hops"] = sr.get("mean_hops", 0)
            metrics["mean_io_time"] = sr.get("mean_io_time", 0)
            metrics["mean_cpus"] = sr.get("mean_cpu_time", 0)
            metrics["latency_95"] = sr.get("p95_latency", 0)
            break

    # Override with span metrics if available
    span_name = f"search-with-L={search_l}-bw={beam_width}"
    for span in search.get("span_metrics", {}).get("spans", []):
        if span.get("span_name") == span_name:
            data = span.get("metrics", {})
            for key in ("qps", "recall", "mean_latency", "mean_ios", "mean_comps",
                        "mean_hops", "mean_io_time", "mean_cpus"):
                if key in data:
                    metrics[key] = data[key]
            break

    return metrics


def compute_diff(baseline_json: list[dict], target_json: list[dict]) -> list[dict]:
    """
    Compare baseline and target JSONs.
    Returns a flat list of metric diffs:
        [{category, metric, baseline, target, deviation}, ...]
    """
    rows = []

    for baseline, target in zip(baseline_json, target_json):
        b_results = baseline.get("results", {})
        t_results = target.get("results", {})

        inp = target.get("input", {})
        search_phase = inp.get("content", {}).get("search_phase", {})
        search_list = search_phase.get("search_list", [2000])
        beam_width = search_phase.get("beam_width", 4)
        primary_l = search_list[0] if search_list else 2000

        # Build metrics
        b_build = extract_build_metrics(b_results)
        t_build = extract_build_metrics(t_results)

        for key in ("total_time", "pq_construction_time", "inmem_index_build_time", "disk_layout_time"):
            if key in t_build or key in b_build:
                bv = b_build.get(key, 0)
                tv = t_build.get(key, 0)
                rows.append({
                    "category": "index-build statistics",
                    "metric": key,
                    "baseline": bv,
                    "target": tv,
                    "deviation": ((tv - bv) / bv * 100) if bv else 0,
                })

        # Search metrics
        b_search = extract_search_metrics(b_results, primary_l, beam_width)
        t_search = extract_search_metrics(t_results, primary_l, beam_width)
        span_cat = f"search-with-L={primary_l}-bw={beam_width}"

        for key in ("qps", "recall", "mean_latency", "latency_95", "mean_ios",
                     "mean_comps", "mean_hops", "mean_io_time", "mean_cpus"):
            if key in t_search or key in b_search:
                bv = b_search.get(key, 0)
                tv = t_search.get(key, 0)
                rows.append({
                    "category": span_cat,
                    "metric": key,
                    "baseline": bv,
                    "target": tv,
                    "deviation": ((tv - bv) / bv * 100) if bv else 0,
                })

    return rows


# =============================================================================
# Thresholds
# =============================================================================

# Format: [max_deviation_%, direction, contract_value]
#   direction: 'GT' = higher is better, 'LT' = lower is better
#   contract_value: absolute limit (empty string = none)
THRESHOLDS: dict[str, dict[str, list]] = {
    "DiskIndexBuild-PqConstruction": {
        "duration_seconds": [10, "LT", ""],
        "peak_memory_usage": [10, "LT", ""],
    },
    "DiskIndexBuild-InmemIndexBuild": {
        "duration_seconds": [10, "LT", ""],
        "peak_memory_usage": [10, "LT", ""],
    },
    "search_disk_index-search_completed": {
        "duration_seconds": [10, "LT", ""],
        "peak_memory_usage": [10, "LT", 1.42],
    },
    "disk_index_perf_test": {
        "total_duration_seconds": [10, "LT", ""],
    },
    "index-build statistics": {
        # Calibrated from 5 GitHub runner runs (10 observations):
        #   Wikipedia: 35.9–37.2s, OpenAI: 23.0–76.4s (SQ_1_2.0 variance)
        #   Contract: worst × 1.5 to absorb shared-runner variance
        "total_time": [10, "LT", 115],
        "total_comparisons": [1, "LT", ""],
        "search_hops": [1, "LT", ""],
    },
    "search-with-L=2000-bw=4": {
        # Calibrated from 5 GitHub runner runs (10 observations)
        "latency_95": [10, "LT", ""],
        "mean_latency": [10, "LT", ""],
        "mean_io_time": [10, "LT", ""],
        "mean_cpus": [15, "LT", ""],   # wider — CPU time is noisy on shared runners
        "qps": [10, "GT", 6.5],
        "mean_ios": [1, "LT", 2410],
        "mean_comps": [1, "LT", 33200],
        "mean_hops": [1, "LT", ""],
        "recall": [1, "GT", 98.0],
    },
    "search-with-L=100-bw=4": {
        "latency_95": [10, "LT", ""],
        "mean_latency": [10, "LT", ""],
        "mean_io_time": [10, "LT", ""],
        "mean_cpus": [15, "LT", ""],
        "qps": [10, "GT", ""],
        "mean_ios": [10, "LT", ""],
        "mean_comps": [10, "LT", ""],
        "mean_hops": [10, "LT", ""],
        "recall": [1, "GT", ""],
    },
    "search-with-L=200-bw=4": {
        "latency_95": [10, "LT", ""],
        "mean_latency": [10, "LT", ""],
        "mean_io_time": [10, "LT", ""],
        "mean_cpus": [15, "LT", ""],
        "qps": [10, "GT", ""],
        "mean_ios": [10, "LT", ""],
        "mean_comps": [10, "LT", ""],
        "mean_hops": [10, "LT", ""],
        "recall": [1, "GT", ""],
    },
}


def allowed_range(threshold: float, direction: str, mode: str) -> tuple[float, float]:
    """Acceptable change range (in %)."""
    if mode == "aa":
        return (-threshold, threshold)
    if direction == "GT":
        return (-threshold, float("inf"))
    return (float("-inf"), threshold)


def fmt_range(lo: float, hi: float) -> str:
    lo_s = "-inf" if lo == float("-inf") else f"{lo}%"
    hi_s = "inf" if hi == float("inf") else f"{hi}%"
    return f"({lo_s} – {hi_s})"


def check_contract(value: float, contract: Any, direction: str) -> tuple[bool, str]:
    """Check if value violates a hard contract. Returns (broken, formatted_contract)."""
    if contract == "":
        return False, "N/A"
    contract = float(contract)
    if direction == "GT" and value < contract:
        return True, f"> {contract}"
    if direction == "LT" and value > contract:
        return True, f"< {contract}"
    return False, str(contract)


# =============================================================================
# Validation
# =============================================================================

def validate(diffs: list[dict], mode: str, run_id: str | None) -> tuple[bool, str]:
    """
    Check all diffs against thresholds.
    Returns (has_failures, markdown_report).
    """
    failed_rows: list[str] = []

    for d in diffs:
        cat, metric = d["category"], d["metric"]
        if cat not in THRESHOLDS or metric not in THRESHOLDS[cat]:
            continue

        pct, direction, contract = THRESHOLDS[cat][metric]
        rng = allowed_range(pct, direction, mode)
        dev = d["deviation"]

        threshold_failed = dev < rng[0] or dev > rng[1]
        contract_broken, contract_fmt = check_contract(d["target"], contract, direction)

        if threshold_failed:
            print(f"THRESHOLD FAILED: {cat}/{metric} change={dev:.2f}% allowed={fmt_range(*rng)}")
        if contract_broken:
            print(f"CONTRACT BROKEN:  {cat}/{metric} value={d['target']} required={contract_fmt}")

        if threshold_failed or contract_broken:
            outcome = []
            if threshold_failed:
                outcome.append("Regression detected")
            if contract_broken:
                outcome.append("Contract broken")
            failed_rows.append(
                f"| {cat}/{metric} | {d['baseline']:.4g} | {d['target']:.4g} | "
                f"{contract_fmt} | {dev:.2f}% | {fmt_range(*rng)} | {', '.join(outcome)} |"
            )

    if not failed_rows:
        return False, ""

    logs_link = ""
    if run_id:
        repo = os.getenv("GITHUB_REPOSITORY", "microsoft/DiskANN")
        logs_link = f"https://github.com/{repo}/actions/runs/{run_id}"

    report = "### ❌ Benchmark Check Failed\n\n"
    if logs_link:
        report += f"Please investigate the [workflow logs]({logs_link}) to determine if the failure is due to your changes.\n\n"
    report += "| Metric | Baseline | Current | Contract | Change | Allowed | Outcome |\n"
    report += "|--------|----------|---------|----------|--------|---------|--------|\n"
    report += "\n".join(failed_rows)

    return True, report


# =============================================================================
# Markdown output
# =============================================================================

def diffs_to_markdown(diffs: list[dict], title: str) -> str:
    """Render diffs as a Markdown table."""
    lines = [
        f"### {title}",
        "",
        "| Category | Metric | Baseline | Current | Change |",
        "|----------|--------|----------|---------|--------|",
    ]
    for d in diffs:
        lines.append(
            f"| {d['category']} | {d['metric']} | {d['baseline']:.4g} | "
            f"{d['target']:.4g} | {d['deviation']:+.2f}% |"
        )
    return "\n".join(lines)


# =============================================================================
# GitHub helpers (stdlib only — no requests dependency)
# =============================================================================

def post_pr_comment(body: str) -> bool:
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    pr = os.getenv("GITHUB_PR_NUMBER")
    if not all([token, repo, pr]):
        print("WARNING: Missing GitHub env vars for PR comment "
              f"(TOKEN={'set' if token else 'missing'}, REPO={repo or 'missing'}, PR={pr or 'missing'})")
        return False

    url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"
    data = json.dumps({"body": body}).encode()
    req = Request(url, data=data, method="POST", headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req, timeout=30) as resp:
            if resp.status < 300:
                print(f"Posted comment to PR #{pr}")
                return True
    except URLError as e:
        print(f"ERROR posting PR comment: {e}")
    return False


def write_step_summary(content: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(content + "\n")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark JSONs, validate thresholds, output Markdown."
    )
    parser.add_argument("--mode", choices=["aa", "pr"], default="aa",
                        help="aa = symmetric thresholds, pr = directional")
    parser.add_argument("--baseline", required=True, help="Baseline JSON path")
    parser.add_argument("--target", required=True, help="Target JSON path")
    parser.add_argument("--title", default="Benchmark Results",
                        help="Title for the Markdown summary table")
    parser.add_argument("--no-comment", action="store_true",
                        help="Skip posting PR comment on failure")
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Baseline: {args.baseline}")
    print(f"Target:   {args.target}")

    baseline = load_json(args.baseline)
    target = load_json(args.target)

    if len(baseline) != len(target):
        print(f"ERROR: JSON arrays differ in length: {len(baseline)} vs {len(target)}")
        return 1

    # Compare
    diffs = compute_diff(baseline, target)
    print(f"\nCompared {len(diffs)} metrics")

    # Write Markdown summary
    md = diffs_to_markdown(diffs, args.title)
    write_step_summary(md)

    # Validate thresholds
    run_id = os.getenv("GITHUB_RUN_ID")
    has_failures, report = validate(diffs, args.mode, run_id)

    if has_failures:
        print("\n" + report)
        write_step_summary(report)
        if args.mode == "pr" and not args.no_comment:
            post_pr_comment(report)
        return 1

    print("\n✅ All metrics within thresholds")
    write_step_summary("### ✅ Benchmark Check Passed\n\nAll metrics within acceptable thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
