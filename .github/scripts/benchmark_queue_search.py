#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Compare input-selected queue searches with #1600 measurement loops."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "diskann/src/neighbor/queue.rs"
SEARCH = ROOT / "diskann/src/neighbor/queue_search.rs"
SEARCH_CONFIG = ROOT / "diskann-benchmark/example/neighbor-queue-search-strategy.json"
BASE = "a58bd09808915b3f457008217cb369a201677fd0"
VARIANTS = ("baseline", "optimized")
METHODS = ("insert", "batch_insert")
REPEATS = 3
BENCHMARK = "diskann-benchmark/src/neighbor_queue.rs"
INPUT = "diskann-benchmark/example/neighbor-queue.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original(path):
    return subprocess.check_output(
        ["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True, encoding="utf-8"
    ).replace("\r\n", "\n")


def check_benchmark():
    if (ROOT / INPUT).read_text(encoding="utf-8") != original(INPUT):
        raise ValueError("Original benchmark input differs from #1600")
    current = (ROOT / BENCHMARK).read_text(encoding="utf-8")
    baseline = original(BENCHMARK)
    for signature in ("fn reset(", "fn fixture(", "fn measure(", "fn run_case("):
        selected = signature.replace("fixture(", "fixture_with_search(")
        actual = function_body(current, selected)
        expected = function_body(baseline, signature)
        actual = actual.replace("Queue::new(capacity).with_insertion_search(search)",
                                "Queue::new(capacity)")
        actual = actual.replace("fixture_with_search(case, rng, input.insertion_search)",
                                "fixture(case, rng)")
        if actual != expected:
            raise ValueError(f"Original benchmark measurement changed: {signature}")


def function_body(source, signature):
    start = source.index("{", source.index(signature))
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def command(args, log):
    with log.open("w", encoding="utf-8") as output:
        output.write(json.dumps([str(arg) for arg in args]) + "\n")
        output.flush()
        subprocess.run(args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)


def build(args):
    check_benchmark()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workload = args.input_file.read_bytes()
    workload_path = output / "workload-input.json"
    workload_path.write_bytes(workload)
    manifest = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "base": BASE,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "runner": os.environ.get("RUNNER_NAME"),
        "rustflags": os.environ.get("RUSTFLAGS"),
        "workload": {"path": str(args.input_file), "sha256": sha256(workload_path)},
        "variants": {},
        "benchmark_sha256": {},
    }
    command(["rustc", "--version", "--verbose"], output / "rustc.txt")
    command(["git", "diff", "HEAD", "--"], output / "working.diff")
    for path in (BENCHMARK, INPUT, "diskann/src/neighbor/queue.rs",
                 "diskann/src/neighbor/queue_search.rs", ".cargo/config.toml", "Cargo.toml",
                 "Cargo.lock", "rust-toolchain.toml"):
        destination = output / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / path).read_bytes())
        manifest["benchmark_sha256"][path] = sha256(destination)
    config = json.loads(args.search_config.read_text(encoding="utf-8"))
    (output / "search-strategy.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    command(
        ["cargo", "test", "--locked", "--release", "-p", "diskann", "--lib",
         "neighbor::queue", "--features", "experimental_diversity_search"],
        output / "tests.log",
    )
    command(
        ["cargo", "build", "--locked", "--release", "-p", "diskann-benchmark", "-vv"],
        output / "build.log",
    )
    suffix = ".exe" if os.name == "nt" else ""
    executable = output / ("diskann-benchmark" + suffix)
    executable.write_bytes(
        (ROOT / "target/release" / ("diskann-benchmark" + suffix)).read_bytes()
    )
    executable.chmod(0o755)
    for variant in VARIANTS:
        content = json.loads(workload)
        if variant == "optimized" and config != {"type": "linear"}:
            content["jobs"][0]["content"]["insertion_search"] = config
        path = output / f"{variant}-input.json"
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        manifest["variants"][variant] = {
            "binary": executable.name, "binary_sha256": sha256(executable),
            "input": path.name, "input_sha256": sha256(path),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["variants"]) != VARIANTS:
        raise ValueError("Expected exactly the original and configured search inputs")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with (output / "runs.jsonl").open("x", encoding="utf-8") as ledger:
        for repeat in range(REPEATS):
            ordered = VARIANTS[repeat % 2:] + VARIANTS[:repeat % 2]
            for variant in ordered:
                info = manifest["variants"][variant]
                input_path = output / info["input"]
                if sha256(input_path) != info["input_sha256"]:
                    raise ValueError("Benchmark input changed after build")
                executable = output / info["binary"]
                if sha256(executable) != info["binary_sha256"]:
                    raise ValueError(f"{variant} binary changed after build")
                raw = output / f"raw-{repeat}-{variant}.json"
                if raw.exists():
                    raise FileExistsError(raw)
                invocation = [
                    str(executable), "run", "--input-file", str(input_path),
                    "--output-file", str(raw),
                ]
                start = datetime.now(timezone.utc).isoformat()
                command(invocation, raw.with_suffix(".log"))
                ledger.write(json.dumps({
                    "variant": variant, "repeat": repeat, "raw": raw.name,
                    "input_sha256": info["input_sha256"],
                    "binary_sha256": info["binary_sha256"], "head": manifest["head"],
                    "started": start, "finished": datetime.now(timezone.utc).isoformat(),
                    "command": invocation,
                    "affinity": sorted(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity") else None,
                }) + "\n")
                ledger.flush()
                print(raw.name, flush=True)


def summarize(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    entries = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    expected = {(variant, repeat) for variant in VARIANTS for repeat in range(REPEATS)}
    if (len(entries) != len(expected)
            or {(e["variant"], e["repeat"]) for e in entries} != expected):
        raise ValueError("Incomplete or duplicate process set")
    cases = None
    groups, signatures, records = {}, {}, []
    for entry in entries:
        variant, repeat = entry["variant"], entry["repeat"]
        info = manifest["variants"][variant]
        input_path = output / info["input"]
        if (sha256(input_path) != info["input_sha256"]
                or entry["input_sha256"] != info["input_sha256"]
                or entry["binary_sha256"] != info["binary_sha256"]
                or entry["head"] != manifest["head"]):
            raise ValueError("Recorded source, input, or executable hash differs")
        job_input = json.loads(input_path.read_text(encoding="utf-8"))["jobs"][0]
        if cases is None:
            cases = job_input["content"]["cases"]
        elif cases != job_input["content"]["cases"]:
            raise ValueError("Search inputs use different cases")
        raw = json.loads((output / entry["raw"]).read_text(encoding="utf-8-sig"))
        if len(raw) != 1 or raw[0]["input"] != job_input:
            raise ValueError("Recorded input differs from the original benchmark")
        results = raw[0]["results"]
        if [result["case"] for result in results] != cases:
            raise ValueError("Missing or reordered benchmark cases")
        for case_index, result in enumerate(results):
            signature = {k: v for k, v in result.items() if k != "samples"}
            if signature != signatures.setdefault(case_index, signature):
                raise ValueError("Fixture work counts differ between processes")
            samples = result["samples"]
            if len(samples) != job_input["content"]["samples"]:
                raise ValueError("Wrong number of benchmark samples")
            values = {method: [] for method in METHODS}
            for round_index, sample in enumerate(samples):
                first = (round_index + case_index) % 2
                methods = list(METHODS[first:] + METHODS[:first])
                if (sample["round"] != round_index or sample["first_method"] != methods[0]
                        or [t["method"] for t in sample["timings"]] != methods):
                    raise ValueError("Original method order changed")
                for timing in sample["timings"]:
                    value = timing["ns_per_update"]
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"Invalid timing: {entry['raw']}: {timing}")
                    values[timing["method"]].append(value)
            for method, timings in values.items():
                record = {
                    **entry, "case": result["case"], "method": method,
                    "median_ns": statistics.median(timings),
                    "sample_min_ns": min(timings), "sample_max_ns": max(timings),
                }
                records.append(record)
                groups.setdefault((case_index, method), {}).setdefault(variant, {})[repeat] = record
    comparisons = []
    for (case_index, method), variants in groups.items():
        before = [variants["baseline"][r]["median_ns"] for r in range(REPEATS)]
        after = [variants["optimized"][r]["median_ns"] for r in range(REPEATS)]
        ratios = [b / c for b, c in zip(before, after)]
        spread = max((max(v) - min(v)) / statistics.median(v) for v in (before, after))
        gain = statistics.median(before) / statistics.median(after)
        direction = "gain" if min(ratios) > 1 and gain - 1 > spread else (
            "loss" if max(ratios) < 1 and 1 - gain > spread else "inconclusive"
        )
        comparisons.append({
            "case": cases[case_index], "method": method,
            "baseline_ns": before, "candidate_ns": after, "paired_speedups": ratios,
            "speedup": gain, "relative_process_range": spread, "repeatability": direction,
        })
    for name, data in (("process-medians", records), ("comparisons", comparisons)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [
        "### Four queue-update groups: selected workload",
        "",
        f"Source: `{manifest['head']}`; workload reference: `{manifest['base']}`.",
        "One executable measures insert and batch_insert with each input-selected search.",
        "Fixture generation, reset, timing loops, and method order match #1600.",
        "Cases and repetitions come from the selected input.",
        "No lookup-only suite, generated cases, population sweep, or repetition cap.",
        "Speedup = original / lookup-optimized, comparing the same update method.",
        "Each cell lists three independent process medians, in ns/update.",
        "Search configuration execution order alternates; compilation finishes before measurement.",
        "The screen requires consistent paired direction exceeding process range;",
        "it is not a confidence interval. Historical #1600 timings are not pooled.",
        "",
        "| Case | Method | Original ns/update | Optimized ns/update | Speedup | Process range | Screen |",
        "|" + "---|" * 7,
    ]
    for c in comparisons:
        lines.append("| " + " | ".join([
            "`" + json.dumps(c["case"], separators=(",", ":")) + "`", c["method"],
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
    parser.add_argument("--input-file", type=Path, default=ROOT / INPUT)
    parser.add_argument("--search-config", type=Path, default=SEARCH_CONFIG)
    args = parser.parse_args()
    {"build": build, "run": run, "summarize": summarize}[args.operation](args)


if __name__ == "__main__":
    main()
