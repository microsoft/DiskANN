# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from summarize_neighbor_queue import main


class SummaryTests(unittest.TestCase):
    def summarize(self, insert, batch):
        case = {
            "capacity": 20,
            "initial_size": 20,
            "input_size": 20,
            "initial_cutoff_survivors": 0,
            "placement": "uniform",
            "order": "shuffled",
        }
        data = [{
            "input": {},
            "results": [{
                "case": case,
                "samples": [
                    {"timings": [
                        {"method": "insert", "ns_per_update": a},
                        {"method": "batch_insert", "ns_per_update": b},
                    ]}
                    for a, b in zip(insert, batch)
                ],
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main([path, path])
            return output.getvalue().splitlines()[-1].split("|")[1:-1]

    def test_retains_zero_samples_in_median(self):
        cells = self.summarize([10, 10, 10], [0, 0, 30])
        self.assertEqual(cells[8].strip(), "0.000")
        self.assertEqual(cells[9].strip(), "N/A")

    def test_zero_insert_median_has_no_ratio(self):
        for batch in [0, 10]:
            with self.subTest(batch=batch):
                cells = self.summarize([0], [batch])
                self.assertEqual(cells[9].strip(), "N/A")

    def test_positive_medians_keep_ratio(self):
        cells = self.summarize([100], [50])
        self.assertEqual(cells[9].strip(), "0.50x")

    def test_rejects_negative_and_nonfinite_timings(self):
        for value in [-1, float("nan"), float("inf"), -float("inf")]:
            for method in ["insert", "batch"]:
                with self.subTest(value=value, method=method):
                    with self.assertRaisesRegex(ValueError, "invalid timings"):
                        self.summarize(
                            [value if method == "insert" else 10],
                            [value if method == "batch" else 10],
                        )


if __name__ == "__main__":
    unittest.main()
