#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Summarize repeated neighbor-queue benchmark runs as Markdown."""

import json
import math
import statistics
import sys
from pathlib import Path


def main(paths):
    if len(paths) < 2:
        raise ValueError("Provide at least two benchmark result files")
    runs = [json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in paths]
    if any(len(run) != 1 for run in runs):
        raise ValueError("Each result file must contain one benchmark job")
    jobs = [run[0] for run in runs]
    first = jobs[0]
    if any(job["input"] != first["input"] for job in jobs):
        raise ValueError("Benchmark inputs differ")
    if any(len(job["results"]) != len(first["results"]) for job in jobs):
        raise ValueError("Benchmark case counts differ")

    methods = ("insert", "batch_insert")
    print("### Neighbor queue: insert vs batch_insert")
    print()
    print(
        "Times are µs per queue update. Each value is the median across all "
        f"samples from {len(paths)} independent processes; the final columns "
        "show each process median. Methods share fixtures and rotate order. "
        "Runners are shared; these results do not measure end-to-end search "
        "performance."
    )
    print()
    print(
        "| # | L | Initial | Candidates | Cutoff survivors | Placement | Order | "
        "Insert | Batch insert | Batch insert / insert | "
        "Insert run medians | Batch insert run medians |"
    )
    print("|" + "---|" * 12)

    for index, result in enumerate(first["results"]):
        case = result["case"]
        samples = {method: [] for method in methods}
        medians = {method: [] for method in methods}
        for job in jobs:
            current = job["results"][index]
            if current["case"] != case:
                raise ValueError(f"Case {index + 1} differs between runs")
            for sample in current["samples"]:
                names = [timing["method"] for timing in sample["timings"]]
                if sorted(names) != sorted(methods):
                    raise ValueError(f"Case {index + 1} has missing or duplicate methods")
            for method in methods:
                values = [
                    next(t["ns_per_update"] for t in sample["timings"] if t["method"] == method)
                    / 1000
                    for sample in current["samples"]
                ]
                if not values:
                    raise ValueError(f"Case {index + 1} has no samples for {method}")
                if any(not math.isfinite(value) or value <= 0 for value in values):
                    raise ValueError(f"Case {index + 1} has invalid timings for {method}")
                samples[method].extend(values)
                medians[method].append(statistics.median(values))
        total = {method: statistics.median(values) for method, values in samples.items()}
        cells = [
            index + 1,
            case["capacity"],
            case["initial_size"],
            case["input_size"],
            case["initial_cutoff_survivors"],
            case["placement"],
            case["order"],
            f'{total["insert"]:.3f}',
            f'{total["batch_insert"]:.3f}',
            f'{total["batch_insert"] / total["insert"]:.2f}x',
            " / ".join(f"{value:.3f}" for value in medians["insert"]),
            " / ".join(f"{value:.3f}" for value in medians["batch_insert"]),
        ]
        print("| " + " | ".join(str(cell) for cell in cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
