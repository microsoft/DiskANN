# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Summarize complete ABBA blocks without dropping slow or differing outputs."""

import argparse
from collections import defaultdict
import copy
import itertools
import json
from pathlib import Path
import statistics

from run_ivf_abba import validate_run
from run_ivf_experiment import sha256


BUILD_TIMES = ["two_means_seconds", "reassignment_seconds", "other_ivf_seconds",
               "ivf_seconds", "centroid_graph_seconds", "total_seconds"]
BUILD_WORK = ["inserted", "splits", "split_points", "split_parent_histogram",
              "lists", "empty_lists", "max_list_size", "index_fingerprint"]
SEARCH_WORK = ["k", "requested_nprobe", "effective_nprobe", "queries", "recall_at_k",
               "mean_points_scanned", "mean_lists_scanned", "mean_centroid_comparisons",
               "mean_results_returned", "result_fingerprint"]


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values), "samples": values}


def compare(blocks, extract):
    a, b, pairs = [], [], []
    for block in blocks:
        a_values = [value for index in [0, 3] for value in extract(block[index])]
        b_values = [value for index in [1, 2] for value in extract(block[index])]
        a.extend(a_values)
        b.extend(b_values)
        pairs.append([statistics.mean(a_values), statistics.mean(b_values)])
    baseline, candidate = statistics.mean(a), statistics.mean(b)
    reductions = [100 * (1 - right / left) if left else None for left, right in pairs]
    # With four blocks there are just 4**4 possible ordered bootstrap draws.
    # Enumerate them instead of making a small-sample interval seem more precise
    # by repeatedly sampling the same few observations.
    interval = None
    if len(pairs) >= 2 and len(pairs) <= 5 and all(left > 0 for left, _ in pairs):
        draws = sorted(
            100 * (1 - sum(pairs[i][1] for i in draw) / sum(pairs[i][0] for i in draw))
            for draw in itertools.product(range(len(pairs)), repeat=len(pairs))
        )
        interval = [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
    return {"A": distribution(a), "B": distribution(b),
            "reduction_pct": 100 * (1 - candidate / baseline) if baseline else None,
            "block_means_A_B": pairs, "block_reductions_pct": reductions,
            "blocks_faster": sum(right < left for left, right in pairs),
            "block_bootstrap_percentile95": interval}


def distinct(records, fields):
    values = {json.dumps({key: row[key] for key in fields}, sort_keys=True) for row in records}
    return [json.loads(value) for value in sorted(values)]


def portable_input(spec):
    raw = json.loads(spec)
    raw["search_directories"] = ["<dataset-directory>"]
    dataset = raw["jobs"][0]["content"]["dataset"]
    for key in ["data", "queries", "groundtruth"]:
        if dataset.get(key):
            dataset[key] = dataset[key].replace("\\", "/").split("/")[-1]
    return raw


def summarize(directory, allow_partial):
    protocol = json.loads((directory / "protocol.json").read_text())
    runs, missing = {}, []
    for entry in protocol["plan"]:
        name = entry["name"]
        path = directory / (name + ".json")
        meta_path = path.with_suffix(".metadata.json")
        if not meta_path.exists():
            missing.append(name)
            continue
        result = json.loads(path.read_text())
        meta = json.loads(meta_path.read_text())
        source = directory / (name + "-input.json")
        validate_run(result, meta, entry["input"], protocol["executable_sha256"],
                     sha256(source), protocol["shapes"][entry["dataset"]], protocol["affinity"])
        if source.read_text(encoding="utf-8") != entry["input"]:
            raise RuntimeError(f"Input no longer matches plan: {name}")
        runs[name] = {
            "name": name, "round": entry["round"], "dataset": entry["dataset"],
            "candidate": entry["candidate"], "position": entry["position"],
            "strategy": entry["strategy"], "results": result[0]["results"],
            "metadata": {key: value for key, value in meta.items() if key not in ["command", "pid"]},
        }
    if missing and not allow_partial:
        raise RuntimeError(f"Only {len(runs)}/{len(protocol['plan'])} invocations complete")

    grouped = defaultdict(list)
    unfinished_blocks = []
    for first in range(0, len(protocol["plan"]), 4):
        entries = protocol["plan"][first:first + 4]
        if len(entries) != 4 or [e["position"] for e in entries] != [0, 1, 2, 3]:
            raise RuntimeError("Malformed ABBA protocol")
        expected = ["paired_point", entries[0]["candidate"], entries[0]["candidate"], "paired_point"]
        if [e["strategy"] for e in entries] != expected:
            raise RuntimeError("Malformed ABBA strategy order")
        if all(e["name"] in runs for e in entries):
            grouped[(entries[0]["dataset"], entries[0]["candidate"])].append([runs[e["name"]] for e in entries])
        else:
            unfinished_blocks.append(entries[0]["name"])

    comparisons = []
    for (dataset, candidate), blocks in grouped.items():
        baseline = [run for block in blocks for run in [block[0], block[3]]]
        candidates = [run for block in blocks for run in [block[1], block[2]]]
        entry = {"dataset": dataset, "candidate": candidate, "complete_blocks": len(blocks),
                 "build": {}, "search": [], "process": {}}
        for field in BUILD_TIMES:
            entry["build"][field] = compare(blocks, lambda run: [b[field] for b in run["results"]["builds"]])
        for field in ["wall_seconds", "user_seconds", "kernel_seconds"]:
            entry["process"][field] = compare(blocks, lambda run: [run["metadata"][field]])
        build_a = distinct([b for run in baseline for b in run["results"]["builds"]], BUILD_WORK)
        build_b = distinct([b for run in candidates for b in run["results"]["builds"]], BUILD_WORK)
        entry["build_work_A"] = build_a
        entry["build_work_B"] = build_b
        entry["identical_build_work"] = build_a == build_b
        probes = sorted({s["requested_nprobe"] for run in baseline for s in run["results"]["searches"]})
        for probe in probes:
            def search_rows(run):
                return [s for s in run["results"]["searches"] if s["requested_nprobe"] == probe]
            work_a = distinct([s for run in baseline for s in search_rows(run)], SEARCH_WORK)
            work_b = distinct([s for run in candidates for s in search_rows(run)], SEARCH_WORK)
            item = {"nprobe": probe, "work_A": work_a, "work_B": work_b,
                    "identical_quality_work": work_a == work_b,
                    "total_seconds": compare(blocks, lambda run: [s["total_seconds"] for s in search_rows(run)]),
                    "mean_query_ns": compare(blocks, lambda run: [s["latency_ns"]["mean"] for s in search_rows(run)])}
            if len(work_a) == len(work_b) == 1:
                a, b = work_a[0], work_b[0]
                item["recall_delta_percentage_points"] = 100 * (b["recall_at_k"] - a["recall_at_k"])
                item["vectors_scanned_change_pct"] = 100 * (b["mean_points_scanned"] / a["mean_points_scanned"] - 1)
            entry["search"].append(item)
        if candidate in ["separate", "packed_paired"]:
            if not entry["identical_build_work"] or not all(s["identical_quality_work"] for s in entry["search"]):
                raise RuntimeError(f"Exact strategy differs from paired: {dataset}/{candidate}")
        comparisons.append(entry)

    public_protocol = copy.deepcopy(protocol)
    for entry in public_protocol["plan"]:
        entry["input"] = portable_input(entry["input"])
    return {
        "complete": not missing, "completed_invocations": len(runs),
        "expected_invocations": len(protocol["plan"]), "missing_invocations": missing,
        "unfinished_blocks": unfinished_blocks, "protocol": public_protocol,
        "comparison_basis": "Each B uses only the two A invocations in its own ABBA block; no sample exclusions",
        "interval_caveat": "Only four blocks per pair; descriptive bootstrap intervals do not model temporal dependence, background activity or multiple comparisons",
        "comparisons": comparisons, "runs": list(runs.values()),
    }


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = summarize(args.directory, args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Complete={report['complete']}; {report['completed_invocations']}/{report['expected_invocations']} invocations")
    for row in report["comparisons"]:
        fit = row["build"]["two_means_seconds"]["reduction_pct"]
        total = row["build"]["total_seconds"]["reduction_pct"]
        print(row["dataset"], row["candidate"], f"blocks={row['complete_blocks']}",
              f"fit={fit:.2f}% total={total:.2f}%", f"same_index={row['identical_build_work']}")


if __name__ == "__main__":
    run()
