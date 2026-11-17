# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Check the four-group experiment without executing timing code."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
