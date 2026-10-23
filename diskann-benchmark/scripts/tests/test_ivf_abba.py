# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Check continuation integrity and denominators when search is discontinued."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_ivf_abba import BUILD_TIMES, summarize
from run_ivf_abba import build_only_input, continue_build_only, dataset_shape, strategy_names, validate_run


EXECUTABLE_HASH = "f" * 64


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def protocol():
    entries = []
    for round_id in range(2):
        for position, strategy in enumerate(["paired_point", "packed_paired", "packed_paired", "paired_point"]):
            spec = {
                "search_directories": ["corpus"],
                "jobs": [{"type": "graph-ivf-build", "content": {
                    "dataset": {"data": "data.bin", "queries": "query.bin", "groundtruth": "gt.bin"},
                    "build": {"repetitions": 1, "seed": 7, "two_means_strategy": strategy},
                    "graph": {"max_degree": 16, "l_build": 64},
                    "search": {"k": 1, "nprobe": [1], "repetitions": 1},
                }}],
            }
            entries.append({"name": f"r{round_id}-{position}-{strategy}", "round": round_id,
                            "dataset": "tiny", "candidate": "packed_paired", "position": position,
                            "strategy": strategy, "input": json.dumps(spec, indent=2) + "\n"})
    return {"executable_sha256": EXECUTABLE_HASH, "affinity": 16, "all_queries": True,
            "shapes": {"tiny": {"data": [4, 1], "queries": [2, 1]}}, "plan": entries}


def record(directory, entry, seconds=10):
    name = entry["name"]
    content = json.loads(entry["input"])["jobs"][0]["content"]
    source = directory / (name + "-input.json")
    source.write_bytes(entry["input"].encode("utf-8"))
    build = {field: seconds for field in BUILD_TIMES}
    build.update(inserted=4, splits=1, split_points=4, split_parent_histogram={"1": 1},
                 lists=2, empty_lists=0, max_list_size=2, index_fingerprint="identical")
    searches = []
    if content.get("search") is not None:
        searches.append({"k": 1, "requested_nprobe": 1, "effective_nprobe": 1, "queries": 2,
                         "recall_at_k": 1.0, "mean_points_scanned": 2.0, "mean_lists_scanned": 1.0,
                         "mean_centroid_comparisons": 2.0, "mean_results_returned": 1.0,
                         "result_fingerprint": "same", "total_seconds": seconds * 10,
                         "latency_ns": {"mean": seconds * 5e9}})
    result = [{"results": {"vectors": 4, "dimensions": 1, "threads": 1,
                           "builds": [build], "searches": searches}}]
    metadata = {"exit_code": 0, "executable_sha256": EXECUTABLE_HASH,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "affinity_observed": 16, "priority_class": 32,
                "wall_seconds": seconds, "user_seconds": seconds, "kernel_seconds": 0.0}
    write_json(directory / (name + ".json"), result)
    write_json(directory / (name + ".metadata.json"), metadata)
    return result, metadata


class BuildOnlyContinuationTests(unittest.TestCase):
    def test_parameterized_candidates_reject_confounded_protocols(self):
        self.assertEqual(strategy_names("paired_point", []), ["tiled_exact"])
        self.assertEqual(strategy_names("original", ["new"]), ["new"])
        for candidates in [["paired_point"], ["a", "a"], ["a/b"], [""]]:
            with self.assertRaises(ValueError):
                strategy_names("paired_point", candidates)

    def test_custom_reference_uses_its_own_complete_abba_block(self):
        for entry in self.protocol["plan"]:
            if entry["strategy"] == "paired_point":
                entry["strategy"] = "custom_baseline"
            spec = json.loads(entry["input"])
            spec["jobs"][0]["content"]["build"]["two_means_strategy"] = entry["strategy"]
            entry["input"] = json.dumps(spec, indent=2) + "\n"
            record(self.original, entry)
        self.protocol["reference"] = "custom_baseline"
        write_json(self.original / "protocol.json", self.protocol)
        report = summarize(self.original, False)
        self.assertTrue(report["complete"])
        self.assertEqual(report["comparisons"][0]["complete_blocks"], 2)

    def test_exact_candidate_rejects_layout_or_search_differences(self):
        for entry in self.protocol["plan"]:
            entry["candidate"] = "tiled_exact"
            if entry["strategy"] == "packed_paired":
                entry["strategy"] = "tiled_exact"
            spec = json.loads(entry["input"])
            spec["jobs"][0]["content"]["build"]["two_means_strategy"] = entry["strategy"]
            entry["input"] = json.dumps(spec, indent=2) + "\n"
            record(self.original, entry)
        write_json(self.original / "protocol.json", self.protocol)
        self.assertTrue(summarize(self.original, False)["complete"])
        path = self.original / (self.protocol["plan"][1]["name"] + ".json")
        saved = json.loads(path.read_text())
        for section, field in [("builds", "index_fingerprint"), ("searches", "result_fingerprint")]:
            changed = copy.deepcopy(saved)
            changed[0]["results"][section][0][field] = "different"
            write_json(path, changed)
            with self.assertRaisesRegex(RuntimeError, "Exact strategy differs"):
                summarize(self.original, False)
        write_json(path, saved)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / "original"
        self.original.mkdir()
        self.continued = self.root / "continued"
        self.continued.mkdir()
        self.protocol = protocol()
        write_json(self.original / "protocol.json", self.protocol)

    def test_no_query_files_are_required_for_build_only(self):
        spec = json.loads(build_only_input(self.protocol["plan"][0]["input"]))
        spec["search_directories"] = [str(self.root)]
        (self.root / "data.bin").write_bytes(struct.pack("<II4f", 4, 1, 1, 2, 3, 4))
        self.assertEqual(dataset_shape(json.dumps(spec)), {"data": [4, 1]})
        before = json.loads(self.protocol["plan"][0]["input"])["jobs"][0]["content"]
        after = spec["jobs"][0]["content"]
        self.assertEqual(before["build"], after["build"])
        self.assertEqual(before["graph"], after["graph"])
        self.assertEqual(after["dataset"], {"data": "data.bin"})
        self.assertNotIn("search", after)

    def test_build_only_validation_rejects_unexpected_search(self):
        entry = copy.deepcopy(self.protocol["plan"][0])
        entry["input"] = build_only_input(entry["input"])
        result, metadata = record(self.continued, entry)
        args = (metadata, entry["input"], EXECUTABLE_HASH, metadata["input_sha256"],
                {"data": [4, 1]}, 16)
        validate_run(result, *args)
        result[0]["results"]["searches"].append({})
        with self.assertRaisesRegex(RuntimeError, "unexpectedly ran search"):
            validate_run(result, *args)
        result[0]["results"]["searches"] = []
        result[0]["results"]["builds"][0]["inserted"] = 3
        with self.assertRaisesRegex(RuntimeError, "Incomplete dataset insertion"):
            validate_run(result, *args)

    def test_prefix_is_immutable_and_mixed_block_keeps_every_build(self):
        record(self.original, self.protocol["plan"][0], 100)
        before = {p.name: p.read_bytes() for p in self.original.iterdir()}
        amended = continue_build_only(self.original, EXECUTABLE_HASH)
        self.assertEqual(amended, continue_build_only(self.original, EXECUTABLE_HASH))
        self.assertEqual(amended["continuation"]["completed_prefix"], 1)
        self.assertEqual(amended["continuation"]["remaining_builds"], 7)
        write_json(self.continued / "protocol.json", amended)
        for entry, seconds in zip(amended["plan"][1:], [1, 1, 2, 4, 3, 3, 4]):
            record(self.continued, entry, seconds)
        report = summarize(self.continued, False)
        comparison = report["comparisons"][0]
        self.assertTrue(report["complete"])
        self.assertEqual(report["completed_invocations"], 8)
        self.assertEqual(comparison["build"]["total_seconds"]["A"]["samples"], [100, 2, 4, 4])
        self.assertEqual(comparison["build"]["total_seconds"]["B"]["samples"], [1, 1, 3, 3])
        self.assertEqual(comparison["build"]["total_seconds"]["A"]["mean"], 27.5)
        self.assertEqual(len(comparison["mixed_mode_blocks"]), 1)
        self.assertEqual(comparison["complete_search_blocks"], 0)
        self.assertEqual(comparison["search"], [])
        self.assertEqual(comparison["process"], {})
        self.assertEqual(comparison["build_by_mode"]["build_only"]["complete_blocks"], 1)
        self.assertEqual(report["runs"][0]["results"]["searches"][0]["queries"], 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.original.iterdir()})

    def test_complete_search_blocks_use_only_search_samples(self):
        for entry, seconds in zip(self.protocol["plan"][:4], [10, 8, 8, 10]):
            record(self.original, entry, seconds)
        amended = continue_build_only(self.original, EXECUTABLE_HASH)
        write_json(self.continued / "protocol.json", amended)
        for entry, seconds in zip(amended["plan"][4:], [4, 3, 3, 4]):
            record(self.continued, entry, seconds)
        comparison = summarize(self.continued, False)["comparisons"][0]
        self.assertEqual(comparison["complete_blocks"], 2)
        self.assertEqual(comparison["complete_search_blocks"], 1)
        self.assertEqual(comparison["search"][0]["total_seconds"]["A"]["samples"], [100, 100])
        self.assertEqual(comparison["process"], {})
        self.assertEqual(set(comparison["process_by_mode"]), {"build_search", "build_only"})
        self.assertEqual(comparison["mixed_mode_blocks"], [])

    def test_unfinished_or_nonconsecutive_original_runs_are_rejected(self):
        path = self.original / (self.protocol["plan"][0]["name"] + ".log")
        path.write_text("still running")
        with self.assertRaisesRegex(RuntimeError, "Unfinished original invocation"):
            continue_build_only(self.original, EXECUTABLE_HASH)
        path.unlink()
        record(self.original, self.protocol["plan"][1])
        with self.assertRaisesRegex(RuntimeError, "consecutive completed prefix"):
            continue_build_only(self.original, EXECUTABLE_HASH)

    def test_reused_artifact_changes_are_rejected(self):
        record(self.original, self.protocol["plan"][0])
        amended = continue_build_only(self.original, EXECUTABLE_HASH)
        write_json(self.continued / "protocol.json", amended)
        output = self.original / (self.protocol["plan"][0]["name"] + ".json")
        output.write_text(output.read_text() + " ")
        with self.assertRaisesRegex(RuntimeError, "Reused artifact changed"):
            summarize(self.continued, True)

    def test_wrong_executable_and_changed_source_protocol_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "original executable"):
            continue_build_only(self.original, "0" * 64)
        amended = continue_build_only(self.original, EXECUTABLE_HASH)
        write_json(self.continued / "protocol.json", amended)
        original = self.original / "protocol.json"
        original.write_text(original.read_text() + " ")
        with self.assertRaisesRegex(RuntimeError, "Original protocol changed"):
            summarize(self.continued, True)


if __name__ == "__main__":
    unittest.main()
