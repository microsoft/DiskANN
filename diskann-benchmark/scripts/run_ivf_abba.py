# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run a resumable, sequential ABBA comparison of IVF two-means strategies."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

from run_ivf_experiment import sha256


DEFAULT_CANDIDATES = ["tiled_exact"]


def strategy_names(reference, candidates):
    """Names are protocol data, not an algorithm enum; old saved runs remain readable."""
    candidates = list(candidates or DEFAULT_CANDIDATES)
    for name in [reference, *candidates]:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            raise ValueError("Strategy names must contain only lowercase letters, digits and underscores")
    if reference in candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Candidates must be unique and different from the reference")
    return candidates


def build_only_input(spec):
    """Keep build settings verbatim while disabling query loading and search."""
    raw = json.loads(spec)
    content = raw["jobs"][0]["content"]
    content.pop("search", None)
    content["dataset"].pop("queries", None)
    content["dataset"].pop("groundtruth", None)
    return json.dumps(raw, indent=2) + "\n"


def specification(template, strategy, repetitions, build_only=False):
    spec = json.loads(template.read_text(encoding="utf-8"))
    if len(spec["jobs"]) != 1:
        raise ValueError("Each dataset template must contain one job")
    job = spec["jobs"][0]
    if job["type"] != "graph-ivf-build":
        raise ValueError("Each job must be graph-ivf-build")
    if not build_only and job["content"].get("search") is None:
        raise ValueError("Each job must include search unless --build-only is set")
    job["content"]["build"].update(
        two_means_strategy=strategy, repetitions=repetitions
    )
    encoded = json.dumps(spec, indent=2) + "\n"
    return build_only_input(encoded) if build_only else encoded


def dataset_shape(spec):
    raw = json.loads(spec)
    dataset = raw["jobs"][0]["content"]["dataset"]
    shapes = {}
    keys = ["data", "queries"] if raw["jobs"][0]["content"].get("search") is not None else ["data"]
    for key in keys:
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
    if "queries" in shapes and shapes["data"][1] != shapes["queries"][1]:
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
    if "input" in result[0]:
        executed = result[0]["input"]["content"]
        if any(executed["build"].get(key) != value for key, value in content["build"].items()):
            raise RuntimeError("Executed build settings/strategy differ from the protocol")
        if executed["graph"] != content["graph"] or executed.get("search") != content.get("search"):
            raise RuntimeError("Executed graph/search settings differ from the protocol")
    values = result[0]["results"]
    if [values["vectors"], values["dimensions"]] != shape["data"] or values["threads"] != 1:
        raise RuntimeError("Dataset shape or thread count differs from the protocol")
    builds = values["builds"]
    if len(builds) != content["build"]["repetitions"]:
        raise RuntimeError("Incomplete build results")
    if any(build["inserted"] != values["vectors"] for build in builds):
        raise RuntimeError("Incomplete dataset insertion")
    searches = values["searches"]
    settings = content.get("search")
    if settings is None:
        if searches:
            raise RuntimeError("Build-only invocation unexpectedly ran search")
        return
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


def result_directory(directory, entry):
    """Reused results stay at their original paths and are never overwritten."""
    return Path(entry.get("source_directory", directory))


def continue_build_only(directory, executable_hash):
    """Freeze a completed prefix and amend only the unstarted ABBA positions."""
    directory = directory.resolve()
    protocol_path = directory / "protocol.json"
    prior = json.loads(protocol_path.read_text())
    if prior["executable_sha256"] != executable_hash:
        raise RuntimeError("Continuation must use the original executable")
    if "continuation" in prior:
        raise RuntimeError("Resume an existing continuation directly, without nesting it")
    protocol = copy.deepcopy(prior)
    completed = 0
    missing_seen = False
    for entry in protocol["plan"]:
        output = directory / (entry["name"] + ".json")
        source = directory / (entry["name"] + "-input.json")
        metadata_path = output.with_suffix(".metadata.json")
        if metadata_path.exists():
            if missing_seen:
                raise RuntimeError("Only a consecutive completed prefix can be continued")
            expected_hash = hashlib.sha256(entry["input"].encode("utf-8")).hexdigest()
            if sha256(source) != expected_hash:
                raise RuntimeError(f"Original input changed: {source}")
            validate_run(json.loads(output.read_text()), json.loads(metadata_path.read_text()),
                         entry["input"], executable_hash, expected_hash,
                         prior["shapes"][entry["dataset"]], prior["affinity"])
            entry.update(source_directory=str(directory), input_sha256=sha256(source),
                         output_sha256=sha256(output), metadata_sha256=sha256(metadata_path))
            completed += 1
        else:
            missing_seen = True
            if source.exists() or output.exists() or output.with_suffix(".log").exists():
                raise RuntimeError(f"Unfinished original invocation must be inspected: {output}")
            entry["input"] = build_only_input(entry["input"])
    if completed == len(protocol["plan"]):
        raise RuntimeError("Original protocol is already complete")
    protocol["all_queries"] = False
    protocol["measurement_scope"] = "build-only continuation; completed search samples retained separately"
    protocol["continuation"] = {
        "source_directory": str(directory), "source_protocol_sha256": sha256(protocol_path),
        "completed_prefix": completed, "remaining_builds": len(protocol["plan"]) - completed,
        "reason": "User requested no search in subsequent experiments.",
        "comparison_caveat": "Build blocks retain their original A/B positions. Search and process timings are grouped only within complete blocks of the same mode. A block crossing the switch is reported explicitly.",
    }
    return protocol


def execute_plan(directory, executable, protocol, stop_after=None):
    runner = Path(__file__).with_name("run_ivf_experiment.py")
    executable_hash = sha256(executable)
    if executable_hash != protocol["executable_sha256"]:
        raise RuntimeError("Executable differs from saved protocol")
    continuation = protocol.get("continuation")
    if continuation:
        source_protocol = Path(continuation["source_directory"]) / "protocol.json"
        if sha256(source_protocol) != continuation["source_protocol_sha256"]:
            raise RuntimeError("Original protocol changed after continuation")
    invoked = 0
    for index, entry in enumerate(protocol["plan"]):
        saved = result_directory(directory, entry)
        source = saved / (entry["name"] + "-input.json")
        output = saved / (entry["name"] + ".json")
        metadata_path = output.with_suffix(".metadata.json")
        input_hash = hashlib.sha256(entry["input"].encode("utf-8")).hexdigest()
        if "source_directory" in entry:
            for path, key in [(source, "input_sha256"), (output, "output_sha256"),
                              (metadata_path, "metadata_sha256")]:
                if sha256(path) != entry[key]:
                    raise RuntimeError(f"Reused artifact changed: {path}")
        if source.exists():
            if sha256(source) != input_hash:
                raise RuntimeError(f"Input changed: {source}")
        else:
            source.write_bytes(entry["input"].encode("utf-8"))
        if metadata_path.exists():
            validate_run(json.loads(output.read_text()), json.loads(metadata_path.read_text()),
                         entry["input"], executable_hash, input_hash,
                         protocol["shapes"][entry["dataset"]], protocol["affinity"])
            continue
        if output.exists() or output.with_suffix(".log").exists():
            raise RuntimeError(f"Incomplete invocation must be inspected before resuming: {output}")
        print(f"[{index + 1}/{len(protocol['plan'])}] {entry['name']}", flush=True)
        subprocess.run([
            sys.executable, str(runner), "--exe", str(executable), "--input", str(source),
            "--output", str(output), "--affinity", str(protocol["affinity"]),
        ], check=True)
        validate_run(json.loads(output.read_text()), json.loads(metadata_path.read_text()),
                     entry["input"], executable_hash, input_hash,
                     protocol["shapes"][entry["dataset"]], protocol["affinity"])
        invoked += 1
        if stop_after is not None and invoked >= stop_after:
            print("Stopped at the requested invocation boundary; protocol is resumable.", flush=True)
            return
    print(f"Completed all {len(protocol['plan'])} ABBA build invocations.", flush=True)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=INPUT")
    parser.add_argument("--reference", default="paired_point")
    parser.add_argument("--candidate", action="append", default=[], help="Repeat for each candidate; defaults to tiled_exact")
    parser.add_argument("--build-only", action="store_true", help="Disable search in new invocations")
    parser.add_argument("--continue-from", type=Path, help="Preserve the completed prefix of an existing protocol; requires --build-only")
    parser.add_argument("--prepare-only", action="store_true", help="Save and verify the plan without starting measurements")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--affinity", type=lambda value: int(value, 0), default=16)
    parser.add_argument("--build-repetitions", type=int, default=1)
    parser.add_argument("--stop-after", type=int, help="Stop at an invocation boundary; resume with the same protocol")
    args = parser.parse_args()
    candidates = strategy_names(args.reference, args.candidate)
    if args.rounds < 1 or args.build_repetitions < 1:
        parser.error("rounds and build repetitions must be positive")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after must be positive")
    directory = args.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = args.exe.resolve()
    executable_hash = sha256(executable)
    protocol_file = directory / "protocol.json"
    if args.continue_from:
        if not args.build_only or args.dataset:
            parser.error("--continue-from requires --build-only and no --dataset")
        if args.continue_from.resolve() == directory:
            parser.error("A continuation needs a new directory to preserve the original protocol")
        protocol = continue_build_only(args.continue_from, executable_hash)
        if protocol_file.exists():
            if json.loads(protocol_file.read_text()) != protocol:
                raise RuntimeError("Saved continuation differs from the original prefix")
        else:
            protocol_file.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        if not args.prepare_only:
            execute_plan(directory, executable, protocol, args.stop_after)
        return
    if not args.dataset:
        parser.error("--dataset or --continue-from is required")
    datasets = []
    for item in args.dataset:
        name, separator, filename = item.partition("=")
        if not separator or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            parser.error("Use a lowercase NAME=INPUT for each dataset")
        if name in [entry[0] for entry in datasets]:
            parser.error("Dataset names must be unique")
        datasets.append((name, Path(filename).resolve()))

    plan = []
    for round_id in range(args.rounds):
        # Rotate candidate order between rounds; retain A-B-B-A within each block.
        order = candidates[round_id % len(candidates):] + candidates[:round_id % len(candidates)]
        for dataset, template in datasets:
            for candidate in order:
                for position, strategy in enumerate([args.reference, candidate, candidate, args.reference]):
                    name = f"r{round_id:02}-{dataset}-{candidate}-{position}-{strategy}"
                    plan.append({
                        "name": name, "round": round_id, "dataset": dataset,
                        "candidate": candidate, "position": position, "strategy": strategy,
                        "input": specification(template, strategy, args.build_repetitions, args.build_only),
                    })
    protocol = {
        "schema_version": 2,
        "rounds": args.rounds, "order": "A-B-B-A", "reference": args.reference,
        "candidates": candidates, "executable_sha256": executable_hash,
        "template_sha256": {name: sha256(path) for name, path in datasets},
        "affinity": args.affinity, "build_repetitions": args.build_repetitions,
        "all_queries": not args.build_only, "primary_statistic": "arithmetic mean",
        "outliers": "retain every invocation and every measured build/search pass",
        "shapes": {name: dataset_shape(specification(path, args.reference, args.build_repetitions, args.build_only))
                   for name, path in datasets},
        "plan": plan,
    }
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != protocol:
            raise RuntimeError("Existing protocol differs; use a new output directory")
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    if not args.prepare_only:
        execute_plan(directory, executable, protocol, args.stop_after)


if __name__ == "__main__":
    run()
