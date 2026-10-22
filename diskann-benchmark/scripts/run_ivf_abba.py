# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run a resumable, sequential ABBA comparison of IVF two-means strategies."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

from run_ivf_experiment import sha256


CANDIDATES = [
    "separate", "packed_paired", "batched_4", "batched_8", "sgemm", "center_difference"
]


def specification(template, strategy, repetitions):
    spec = json.loads(template.read_text(encoding="utf-8"))
    if len(spec["jobs"]) != 1:
        raise ValueError("Each dataset template must contain one job")
    job = spec["jobs"][0]
    if job["type"] != "graph-ivf-build" or job["content"].get("search") is None:
        raise ValueError("Each job must include build and search")
    job["content"]["build"].update(
        two_means_strategy=strategy, repetitions=repetitions
    )
    return json.dumps(spec, indent=2) + "\n"


def dataset_shape(spec):
    raw = json.loads(spec)
    dataset = raw["jobs"][0]["content"]["dataset"]
    shapes = {}
    for key in ["data", "queries"]:
        path = Path(dataset[key])
        choices = [path] if path.is_absolute() else [Path(folder) / path for folder in raw["search_directories"]]
        source = next((choice for choice in choices if choice.is_file()), None)
        if source is None:
            raise ValueError(f"Missing {key} file: {path}")
        with source.open("rb") as stream:
            rows, dimensions = struct.unpack("<II", stream.read(8))
        width = 2 if dataset.get("data_type", "float32") == "float16" else 4
        if source.stat().st_size != 8 + rows * dimensions * width:
            raise ValueError(f"Unexpected vector file size/type: {source}")
        shapes[key] = [rows, dimensions]
    if shapes["data"][1] != shapes["queries"][1]:
        raise ValueError("Data and query dimensions differ")
    return shapes


def validate_run(result, metadata, spec, executable_hash, input_hash, shape, affinity):
    if metadata["exit_code"] != 0:
        raise RuntimeError("A failed run must be investigated, not silently replaced")
    if metadata["executable_sha256"] != executable_hash or metadata["input_sha256"] != input_hash:
        raise RuntimeError("Completed run provenance differs from the fixed protocol")
    if metadata["affinity_observed"] != affinity or metadata["priority_class"] != 32:
        raise RuntimeError("Process affinity or priority differs from the protocol")
    if len(result) != 1:
        raise RuntimeError("Expected one result job")
    content = json.loads(spec)["jobs"][0]["content"]
    values = result[0]["results"]
    if [values["vectors"], values["dimensions"]] != shape["data"] or values["threads"] != 1:
        raise RuntimeError("Dataset shape or thread count differs from the protocol")
    builds = values["builds"]
    if len(builds) != content["build"]["repetitions"]:
        raise RuntimeError("Incomplete build results")
    if any(build["inserted"] != values["vectors"] for build in builds):
        raise RuntimeError("Incomplete dataset insertion")
    settings = content["search"]
    searches = values["searches"]
    expected = len(builds) * len(settings["nprobe"]) * settings["repetitions"]
    if len(searches) != expected:
        raise RuntimeError("Incomplete search results")
    if any(search["queries"] != shape["queries"][0] for search in searches):
        raise RuntimeError("Search did not process the complete query file")
    expected_settings = sorted(settings["nprobe"] * (len(builds) * settings["repetitions"]))
    if sorted(search["requested_nprobe"] for search in searches) != expected_settings:
        raise RuntimeError("Search settings differ from the protocol")
    if any(search["mean_results_returned"] != settings["k"] for search in searches):
        raise RuntimeError("Search returned fewer than k results")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=INPUT")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--affinity", type=lambda value: int(value, 0), default=16)
    parser.add_argument("--build-repetitions", type=int, default=1)
    parser.add_argument("--stop-after", type=int, help="Stop at an invocation boundary; resume with the same protocol")
    args = parser.parse_args()
    if args.rounds < 1 or args.build_repetitions < 1:
        parser.error("rounds and build repetitions must be positive")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after must be positive")
    datasets = []
    for item in args.dataset:
        name, separator, filename = item.partition("=")
        if not separator or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            parser.error("Use a lowercase NAME=INPUT for each dataset")
        if name in [entry[0] for entry in datasets]:
            parser.error("Dataset names must be unique")
        datasets.append((name, Path(filename).resolve()))

    directory = args.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = args.exe.resolve()
    executable_hash = sha256(executable)
    plan = []
    for round_id in range(args.rounds):
        # Rotate candidate order between rounds; retain A-B-B-A within each block.
        order = CANDIDATES[round_id % len(CANDIDATES):] + CANDIDATES[:round_id % len(CANDIDATES)]
        for dataset, template in datasets:
            for candidate in order:
                for position, strategy in enumerate(["paired_point", candidate, candidate, "paired_point"]):
                    name = f"r{round_id:02}-{dataset}-{candidate}-{position}-{strategy}"
                    plan.append({
                        "name": name, "round": round_id, "dataset": dataset,
                        "candidate": candidate, "position": position, "strategy": strategy,
                        "input": specification(template, strategy, args.build_repetitions),
                    })
    protocol = {
        "rounds": args.rounds, "order": "A-B-B-A", "reference": "paired_point",
        "candidates": CANDIDATES, "executable_sha256": executable_hash,
        "affinity": args.affinity, "build_repetitions": args.build_repetitions,
        "all_queries": True, "primary_statistic": "arithmetic mean",
        "outliers": "retain every invocation and every measured build/search pass",
        "shapes": {name: dataset_shape(specification(path, "paired_point", args.build_repetitions))
                   for name, path in datasets},
        "plan": plan,
    }
    protocol_file = directory / "protocol.json"
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != protocol:
            raise RuntimeError("Existing protocol differs; use a new output directory")
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    runner = Path(__file__).with_name("run_ivf_experiment.py")
    invoked = 0
    for index, entry in enumerate(plan):
        source = directory / (entry["name"] + "-input.json")
        output = directory / (entry["name"] + ".json")
        metadata_path = output.with_suffix(".metadata.json")
        input_hash = hashlib.sha256(entry["input"].encode("utf-8")).hexdigest()
        if source.exists():
            if sha256(source) != input_hash:
                raise RuntimeError(f"Input changed: {source}")
        else:
            source.write_bytes(entry["input"].encode("utf-8"))
        if metadata_path.exists():
            validate_run(json.loads(output.read_text()), json.loads(metadata_path.read_text()),
                         entry["input"], executable_hash, input_hash, protocol["shapes"][entry["dataset"]], args.affinity)
            continue
        if output.exists() or output.with_suffix(".log").exists():
            raise RuntimeError(f"Incomplete invocation must be inspected before resuming: {output}")
        print(f"[{index + 1}/{len(plan)}] {entry['name']}", flush=True)
        subprocess.run([
            sys.executable, str(runner), "--exe", str(executable), "--input", str(source),
            "--output", str(output), "--affinity", str(args.affinity),
        ], check=True)
        validate_run(json.loads(output.read_text()), json.loads(metadata_path.read_text()),
                     entry["input"], executable_hash, input_hash, protocol["shapes"][entry["dataset"]], args.affinity)
        invoked += 1
        if args.stop_after is not None and invoked >= args.stop_after:
            print("Stopped at the requested invocation boundary; protocol is resumable.", flush=True)
            return
    print(f"Completed all {len(plan)} ABBA invocations.", flush=True)


if __name__ == "__main__":
    run()
