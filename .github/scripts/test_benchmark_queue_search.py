# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Check the four-group experiment without executing timing code."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_queue_search as experiment


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.args = SimpleNamespace(output=self.output)
        self.write_results()

    def write_results(self):
        source = json.loads((experiment.ROOT / experiment.INPUT).read_text())
        job = source["jobs"][0]
        manifest = {
            "head": "head", "base": experiment.BASE, "variants": {},
        }
        for variant in experiment.VARIANTS:
            content = json.loads(json.dumps(source))
            if variant == "optimized":
                content["jobs"][0]["content"]["insertion_search"] = {"type": "binary"}
            path = self.output / f"{variant}-input.json"
            path.write_text(json.dumps(content))
            manifest["variants"][variant] = {
                "binary_sha256": variant, "input": path.name,
                "input_sha256": experiment.sha256(path),
            }
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        entries = []
        for repeat in range(experiment.REPEATS):
            for variant in experiment.VARIANTS:
                info = manifest["variants"][variant]
                job = json.loads((self.output / info["input"]).read_text())["jobs"][0]
                results = []
                for index, case in enumerate(job["content"]["cases"]):
                    values = (100.0, 100.0) if variant == "baseline" else (50.0, 125.0)
                    samples = []
                    for round_index in range(job["content"]["samples"]):
                        first = (round_index + index) % 2
                        methods = experiment.METHODS[first:] + experiment.METHODS[:first]
                        samples.append({
                            "round": round_index, "first_method": methods[0],
                            "timings": [{
                                "method": method,
                                "ns_per_update": values[experiment.METHODS.index(method)],
                            } for method in methods],
                        })
                    results.append({"case": case, "mean_scalar_admissions": 8, "samples": samples})
                raw = self.output / f"raw-{repeat}-{variant}.json"
                raw.write_text(json.dumps([{"input": job, "results": results}]))
                entries.append({
                    "variant": variant, "repeat": repeat, "raw": raw.name, "head": "head",
                    "input_sha256": info["input_sha256"], "binary_sha256": variant,
                })
        self.ledger = self.output / "runs.jsonl"
        self.ledger.write_text("\n".join(json.dumps(entry) for entry in entries))

    def change_raw(self, edit):
        path = self.output / "raw-0-optimized.json"
        data = json.loads(path.read_text())
        edit(data[0])
        path.write_text(json.dumps(data))

    def test_speedups_compare_the_same_update_method(self):
        experiment.summarize(self.args)
        comparisons = json.loads((self.output / "comparisons.json").read_text())
        self.assertEqual(len(comparisons), 46)
        for c in comparisons:
            if c["method"] == "insert":
                self.assertEqual((c["speedup"], c["repeatability"]), (2.0, "gain"))
            else:
                self.assertEqual((c["speedup"], c["repeatability"]), (0.8, "loss"))
            self.assertEqual(len(c["baseline_ns"]), 3)

    def test_rejects_changed_fixture_work(self):
        self.change_raw(lambda job: job["results"][0].update(mean_scalar_admissions=7))
        with self.assertRaisesRegex(ValueError, "work counts"):
            experiment.summarize(self.args)

    def test_rejects_changed_method_order(self):
        self.change_raw(lambda job: job["results"][0]["samples"][0]["timings"].reverse())
        with self.assertRaisesRegex(ValueError, "method order"):
            experiment.summarize(self.args)

    def test_rejects_missing_case(self):
        self.change_raw(lambda job: job["results"].pop())
        with self.assertRaisesRegex(ValueError, "benchmark cases"):
            experiment.summarize(self.args)

    def test_rejects_incomplete_process_set(self):
        self.ledger.write_text("\n".join(self.ledger.read_text().splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, "process set"):
            experiment.summarize(self.args)

    def test_rejects_changed_input(self):
        self.change_raw(lambda job: job["input"]["content"].update(seed=43))
        with self.assertRaisesRegex(ValueError, "input differs"):
            experiment.summarize(self.args)

    def test_benchmark_source_and_input_match_original(self):
        experiment.check_benchmark()

    def test_build_uses_default_or_selected_workload_for_both_searches(self):
        selected = experiment.ROOT / "diskann-benchmark/example/neighbor-queue-large-l.json"
        suffix = ".exe" if experiment.os.name == "nt" else ""
        binary = experiment.ROOT / "target/release" / ("diskann-benchmark" + suffix)
        read_bytes = Path.read_bytes

        def compiled_binary(path):
            return b"test executable" if path == binary else read_bytes(path)

        for index, input_file in enumerate((None, selected)):
            with self.subTest(input_file=input_file):
                output = self.output / f"build-{index}"
                argv = ["benchmark_queue_search.py", "build", "--output", str(output)]
                if input_file is not None:
                    argv.extend(["--input-file", str(input_file)])
                with patch.object(sys, "argv", argv), \
                        patch.object(experiment, "command"), \
                        patch.object(Path, "read_bytes", compiled_binary):
                    experiment.main()

                source = input_file or experiment.ROOT / experiment.INPUT
                self.assertEqual((output / "workload-input.json").read_bytes(),
                                 source.read_bytes())
                expected = json.loads(source.read_bytes())
                for variant in experiment.VARIANTS:
                    if variant == "optimized":
                        expected["jobs"][0]["content"]["insertion_search"] = {"type": "binary"}
                    actual = json.loads((output / f"{variant}-input.json").read_bytes())
                    self.assertEqual(actual, expected)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["workload"]["sha256"], experiment.sha256(source))


if __name__ == "__main__":
    unittest.main()
