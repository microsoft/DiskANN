# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Check the experiment's comparison and rejection rules without timing code."""

import json
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
        self.seed_patch = patch.object(experiment, "SEEDS", (42,))
        self.population_patch = patch.object(experiment, "POPULATIONS", (8,))
        self.seed_patch.start()
        self.population_patch.start()
        self.addCleanup(self.seed_patch.stop)
        self.addCleanup(self.population_patch.stop)
        self.write_results()

    def write_results(self):
        variants = {"simd": 100.0, "binary": 50.0, "hybrid512": 125.0}
        manifest = {"head": "head", "base": "base", "variants": {
            v: {"binary_sha256": v} for v in variants
        }}
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        entries = []
        for repeat in range(3):
            for suite, variant in [
                ("lookup", "simd"), *(("updates", v) for v in variants)
            ]:
                input_path = self.output / f"{suite}-input.json"
                experiment.write_input(input_path, suite, {"seed": 42})
                raw_path = self.output / f"{suite}-{variant}-{repeat}.json"
                samples = []
                for round_index in range(9):
                    if suite == "lookup":
                        for method, value in [("linear", 100.0), ("binary", 50.0)]:
                            samples.append({
                                "round": round_index, "method": method,
                                "ns_per_lookup": value,
                            })
                    else:
                        samples.append({"round": round_index, "blocks": 4, "timings": [
                            {"method": method, "ns_per_update": variants[variant]}
                            for method in ("insert", "batch_insert")
                        ]})
                result = {"samples": samples}
                if suite == "lookup":
                    result.update(length=2000, position="random")
                else:
                    result["case"] = {"capacity": 2000, "initial_size": 2000, "input_size": 8}
                raw_path.write_text(json.dumps([{"input": {"seed": 42}, "results": [result]}]))
                entries.append({
                    "suite": suite, "variant": variant, "seed": 42, "fixtures": 8,
                    "repeat": repeat, "raw": raw_path.name, "input": input_path.name,
                    "input_sha256": experiment.sha256(input_path), "binary_sha256": variant,
                })
        self.ledger = self.output / "runs.jsonl"
        self.ledger.write_text("\n".join(json.dumps(entry) for entry in entries))

    def test_speedups_and_repeatability_use_independent_process_medians(self):
        experiment.summarize(self.args)
        comparisons = json.loads((self.output / "comparisons.json").read_text())
        binary = [c for c in comparisons if c["strategy"] == "binary"]
        hybrid = [c for c in comparisons if c["strategy"] == "hybrid512"]
        self.assertTrue(all(c["speedup"] == 2.0 and c["repeatability"] == "gain" for c in binary))
        self.assertTrue(all(c["speedup"] == 0.8 and c["repeatability"] == "loss" for c in hybrid))
        self.assertTrue(all(len(c["baseline_ns"]) == 3 for c in comparisons))

    def test_rejects_mismatched_actual_blocks(self):
        raw = self.output / "updates-binary-0.json"
        data = json.loads(raw.read_text())
        data[0]["results"][0]["samples"][0]["blocks"] = 3
        raw.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "repetition counts"):
            experiment.summarize(self.args)

    def test_rejects_incomplete_process_set(self):
        self.ledger.write_text("\n".join(self.ledger.read_text().splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, "Incomplete experiment"):
            experiment.summarize(self.args)

    def test_rejects_changed_input(self):
        path = self.output / "updates-input.json"
        data = json.loads(path.read_text())
        data["jobs"][0]["content"]["seed"] = 43
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "input differs"):
            experiment.summarize(self.args)


if __name__ == "__main__":
    unittest.main()
