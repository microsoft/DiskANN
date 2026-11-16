#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Build fixed queue-search variants and compare independent benchmark processes."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "diskann/src/neighbor/queue.rs"
SEARCH = ROOT / "diskann/src/neighbor/queue_search.rs"
SEEDS = (20261114, 9372461)
POPULATIONS = (8, 128)
VARIANTS = ("simd", "binary", "hybrid512")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, log):
    with log.open("w", encoding="utf-8") as output:
        output.write(json.dumps([str(arg) for arg in args]) + "\n")
        output.flush()
        subprocess.run(args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)


def variant_source(current, baseline, variant):
    """Keep the exact original SIMD call context as the full-update baseline."""
    test_marker = "#[cfg(test)]"
    baseline = baseline.split(test_marker, 1)[0] + test_marker + current.split(test_marker, 1)[1]
    if variant == "simd":
        return baseline
    if variant == "insert_hybrid":
        source = baseline.replace(
            "use diskann_wide::{SIMDMask, SIMDPartialOrd, SIMDVector};",
            'use diskann_wide::{SIMDMask, SIMDPartialOrd, SIMDVector};\n'
            '\n#[path = "queue_search.rs"]\nmod search;',
            1,
        )
        old = "let insert_idx = if self.size > 0 {\n            self.get_lower_bound(&nbr)"
        if source.count(old) != 1:
            raise ValueError("Cannot identify the single-item insertion lookup")
        return source.replace(
            old, "let insert_idx = if self.size > 0 {\n"
            "            search::hybrid::<512>(&self.distances, nbr.distance)", 1
        )
    source = current.replace(
        '#[cfg(target_arch = "x86_64")]\n#[path = "queue_search.rs"]',
        '#[path = "queue_search.rs"]',
        1,
    )
    start = source.index("    /// Get the lower bound of the neighbor")
    end = source.index("    /// Get the neighbor at index - SAFETY:", start)
    lookup = "binary" if variant == "binary" else "hybrid::<512>"
    return (
        source[:start]
        + "    /// Return the first distance greater than or equal to the target.\n"
        + "    fn get_lower_bound(&mut self, nbr: &Neighbor<I>) -> usize {\n"
        + f"        search::{lookup}(&self.distances, nbr.distance)\n"
        + "    }\n\n"
        + source[end:]
    )


def build(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    current_bytes = QUEUE.read_bytes()
    current = current_bytes.decode("utf-8").replace("\r\n", "\n")
    baseline = subprocess.check_output(
        ["git", "show", f"{args.base}:diskann/src/neighbor/queue.rs"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).replace("\r\n", "\n")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "head": head,
        "base": args.base,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "runner": os.environ.get("RUNNER_NAME"),
        "rustflags": os.environ.get("RUSTFLAGS"),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "variants": {},
    }
    command(["rustc", "--version", "--verbose"], output / "rustc.txt")
    for config in (".cargo/config.toml", "Cargo.toml", "rust-toolchain.toml"):
        destination = output / "config" / config
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / config).read_bytes())
    try:
        for variant in args.variants:
            directory = output / variant
            directory.mkdir(exist_ok=True)
            source = variant_source(current, baseline, variant)
            QUEUE.write_text(source, encoding="utf-8", newline="\n")
            (directory / "queue.rs").write_text(source, encoding="utf-8")
            (directory / "queue_search.rs").write_bytes(SEARCH.read_bytes())
            command(
                ["cargo", "test", "--locked", "--release", "-p", "diskann", "--lib",
                 "neighbor::queue", "--features", "experimental_diversity_search"],
                directory / "tests.log",
            )
            command(
                ["cargo", "build", "--locked", "--release", "-p", "diskann-benchmark", "-vv"],
                directory / "build.log",
            )
            suffix = ".exe" if os.name == "nt" else ""
            executable = directory / ("diskann-benchmark" + suffix)
            executable.write_bytes(
                (ROOT / "target/release" / ("diskann-benchmark" + suffix)).read_bytes()
            )
            executable.chmod(0o755)
            manifest["variants"][variant] = {
                "binary": executable.name,
                "binary_sha256": sha256(executable),
                "queue_sha256": sha256(directory / "queue.rs"),
                "search_sha256": sha256(directory / "queue_search.rs"),
                "scope": "original SIMD/scalar" if variant == "simd" else
                         "hybrid insert; original SIMD/scalar batch/remove" if variant == "insert_hybrid"
                         else "same search strategy on all architectures (experiment only)",
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        QUEUE.write_bytes(current_bytes)


def cases():
    result = []

    def add(length, count, initial=None, placement="uniform", order="shuffled", survivors=None):
        result.append({
            "capacity": length,
            "initial_size": length if initial is None else initial,
            "input_size": count,
            "initial_cutoff_survivors": count if survivors is None else survivors,
            "placement": placement,
            "order": order,
        })

    for length in (20, 40, 100, 200, 500, 1000, 2000, 8192):
        for count in (1, 8, 2000):
            add(length, count)
    for length in (2000, 8192):
        for count in (2, 4):
            add(length, count)
        for placement in ("head", "middle", "tail"):
            add(length, 8, placement=placement)
        add(length, 8, survivors=0)
    for length in (511, 512, 513):
        add(length, 8)
    for order in ("ascending", "descending"):
        add(2000, 8, order=order)
    for initial in (20, 200, 1000):
        add(2000, 8, initial=initial)
    return result


def write_input(path, tag, content):
    path.write_text(json.dumps({
        "search_directories": [],
        "jobs": [{"type": tag, "content": content}],
    }, indent=2), encoding="utf-8")


def run(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    variants = tuple(manifest["variants"])
    # No compilation overlaps any measurement. Pin all native children to one allowed CPU
    # on Linux; Windows inherits the runner's affinity, recorded in environment.txt.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with (output / "runs.jsonl").open("x", encoding="utf-8") as ledger:
        for seed in SEEDS:
            for population in POPULATIONS:
                lookup_input = output / f"lookup-{seed}-{population}.json"
                update_input = output / f"updates-{seed}-{population}.json"
                write_input(lookup_input, "neighbor-queue-lookup", {
                    "seed": seed, "samples": 9, "fixtures": population,
                    "blocks_per_sample": 32768 // population,
                    "lengths": [20, 40, 100, 200, 500, 511, 512, 513, 1000, 2000, 8192],
                })
                write_input(update_input, "neighbor-queue", {
                    "seed": seed, "samples": 9, "queues_per_sample": population,
                    "blocks_per_sample": 512 // population, "cases": cases(),
                })
                for repeat in range(3):
                    ordered = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
                    jobs = [("lookup", variants[0], lookup_input)]
                    jobs.extend(("updates", variant, update_input) for variant in ordered)
                    for suite, variant, input_path in jobs:
                        info = manifest["variants"][variant]
                        executable = output / variant / info["binary"]
                        if sha256(executable) != info["binary_sha256"]:
                            raise ValueError(f"{variant} binary changed after build")
                        raw = output / f"raw-{suite}-{seed}-{population}-{repeat}-{variant}.json"
                        if raw.exists():
                            raise FileExistsError(raw)
                        invocation = [
                            str(executable), "run", "--input-file", str(input_path),
                            "--output-file", str(raw),
                        ]
                        start = datetime.now(timezone.utc).isoformat()
                        command(invocation, raw.with_suffix(".log"))
                        entry = {
                            "suite": suite, "variant": variant, "seed": seed,
                            "fixtures": population, "repeat": repeat, "raw": raw.name,
                            "input": input_path.name, "input_sha256": sha256(input_path),
                            "binary_sha256": info["binary_sha256"], "head": manifest["head"],
                            "started": start, "finished": datetime.now(timezone.utc).isoformat(),
                            "command": invocation,
                            "affinity": sorted(os.sched_getaffinity(0))
                            if hasattr(os, "sched_getaffinity") else None,
                        }
                        ledger.write(json.dumps(entry) + "\n")
                        ledger.flush()
                        print(raw.name, flush=True)


def summarize(args):
    output = args.output.resolve()
    entries = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    expected = 3 * len(SEEDS) * len(POPULATIONS) * (1 + len(manifest["variants"]))
    if len(entries) != expected:
        raise ValueError(f"Incomplete experiment: {len(entries)} of {expected} processes")
    groups = {}
    records = []
    for entry in entries:
        raw = json.loads((output / entry["raw"]).read_text(encoding="utf-8-sig"))
        if len(raw) != 1:
            raise ValueError("Expected exactly one benchmark job")
        job = raw[0]
        source_input = json.loads((output / entry["input"]).read_text())["jobs"][0]
        if job["input"] != source_input:
            raise ValueError("Recorded input differs from the executed input")
        if sha256(output / entry["input"]) != entry["input_sha256"]:
            raise ValueError("Recorded input hash differs")
        if entry["binary_sha256"] != manifest["variants"][entry["variant"]]["binary_sha256"]:
            raise ValueError("Recorded executable hash differs")
        for result in job["results"]:
            if entry["suite"] == "updates":
                key = json.dumps(result["case"], sort_keys=True)
                samples = {}
                blocks = [sample["blocks"] for sample in result["samples"]]
                for sample in result["samples"]:
                    if len(sample["timings"]) != 2:
                        raise ValueError("Missing full-update method")
                    for timing in sample["timings"]:
                        samples.setdefault(timing["method"], []).append(timing["ns_per_update"])
            else:
                key = json.dumps({"length": result["length"], "position": result["position"]})
                blocks = None
                samples = {}
                for sample in result["samples"]:
                    samples.setdefault(sample["method"], []).append(sample["ns_per_lookup"])
            for method, values in samples.items():
                if len(values) != 9 or any(not math.isfinite(v) or v <= 0 for v in values):
                    raise ValueError(f"Invalid samples: {entry['raw']} {key} {method}")
                record = {
                    **entry, "case": json.loads(key), "method": method,
                    "median_ns": statistics.median(values), "sample_min_ns": min(values),
                    "sample_max_ns": max(values), "blocks": blocks,
                }
                records.append(record)
                strategy = entry["variant"] if entry["suite"] == "updates" else method
                group = (entry["suite"], entry["seed"], entry["fixtures"], key, method
                         if entry["suite"] == "updates" else "lookup")
                groups.setdefault(group, {}).setdefault(strategy, []).append(record)
    comparisons = []
    for group, strategies in groups.items():
        baseline_name = "simd" if group[0] == "updates" else "linear"
        baseline = sorted(strategies[baseline_name], key=lambda r: r["repeat"])
        for strategy, candidate in strategies.items():
            if strategy == baseline_name:
                continue
            candidate.sort(key=lambda r: r["repeat"])
            if len(baseline) != 3 or len(candidate) != 3:
                raise ValueError("Missing independent repeats")
            if [r["blocks"] for r in baseline] != [r["blocks"] for r in candidate]:
                raise ValueError("Variant repetition counts differ")
            before = [r["median_ns"] for r in baseline]
            after = [r["median_ns"] for r in candidate]
            ratios = [b / c for b, c in zip(before, after)]
            spread = max(
                (max(before) - min(before)) / statistics.median(before),
                (max(after) - min(after)) / statistics.median(after),
            )
            gain = statistics.median(before) / statistics.median(after)
            # This is a conservative repeatability screen, not a confidence interval.
            direction = "gain" if min(ratios) > 1 and gain - 1 > spread else (
                "loss" if max(ratios) < 1 and 1 - gain > spread else "inconclusive"
            )
            comparisons.append({
                "suite": group[0], "seed": group[1], "fixtures": group[2],
                "case": json.loads(group[3]), "method": group[4], "strategy": strategy,
                "baseline_ns": before, "candidate_ns": after, "paired_speedups": ratios,
                "speedup": gain, "relative_process_range": spread, "repeatability": direction,
            })
    (output / "process-medians.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    lines = [
        "### Queue insertion search: fixed-source comparison",
        "",
        f"Source: `{manifest['head']}`; exact SIMD baseline: `{manifest['base']}`.",
        "Speedup = SIMD baseline / candidate. Values are ns/lookup or ns/update.",
        "Each cell shows three independent process medians; nine rounds per process.",
        "Two fixed seeds and 8/128 fixture populations are reported separately.",
        "Variant execution order rotates; compilation finishes before measurement.",
        "Lookup kernels rotate within each process. Setup and reset are untimed.",
        "Shared hosted runners are not assumed to be quieter than local hardware.",
        "Repeatability requires a consistent paired direction exceeding the observed",
        "relative process range; it is a screening rule, not a confidence interval.",
        "These warm-fixture microbenchmarks do not measure end-to-end ANN search.",
        "",
        "| Suite | Seed | Fixtures | Case | Method | Strategy | Baseline ns | Candidate ns | Speedup | Process range | Screen |",
        "|" + "---|" * 11,
    ]
    for c in comparisons:
        lines.append("| " + " | ".join([
            c["suite"], str(c["seed"]), str(c["fixtures"]),
            "`" + json.dumps(c["case"], separators=(",", ":")) + "`",
            c["method"], c["strategy"],
            "/".join(f"{v:.2f}" for v in c["baseline_ns"]),
            "/".join(f"{v:.2f}" for v in c["candidate_ns"]),
            f"{c['speedup']:.3f}x", f"{100*c['relative_process_range']:.1f}%",
            c["repeatability"],
        ]) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audited {len(entries)} processes; {len(comparisons)} comparisons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base")
    parser.add_argument("--variants", nargs="+", choices=(*VARIANTS, "insert_hybrid"), default=VARIANTS)
    args = parser.parse_args()
    if args.operation == "build" and not args.base:
        parser.error("build requires the fixed baseline --base SHA")
    {"build": build, "run": run, "summarize": summarize}[args.operation](args)


if __name__ == "__main__":
    main()
