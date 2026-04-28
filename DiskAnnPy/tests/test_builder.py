# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import diskannpy as dap
import numpy as np


class TestBuildDiskIndex(unittest.TestCase):
    def test_valid_shape(self):
        rng = np.random.default_rng(12345)
        rando = rng.random((1000, 100, 5), dtype=np.single)
        with self.assertRaises(ValueError):
            dap.build_disk_index(
                data=rando,
                distance_metric="l2",
                index_directory="test",
                complexity=5,
                graph_degree=5,
                num_of_pq_chunks=1,
                build_memory_maximum=0.01,
                num_threads=1,
                build_pq_bytes=0,
            )

        rando = rng.random((1000, 100, 5), dtype=np.single)
        with self.assertRaises(ValueError):
            dap.build_disk_index(
                data=rando,
                distance_metric="l2",
                index_directory="test",
                complexity=5,
                graph_degree=5,
                num_of_pq_chunks=1,
                build_memory_maximum=0.01,
                num_threads=1,
                build_pq_bytes=5
            )

        rando = rng.random(1000, dtype=np.single)
        with self.assertRaises(ValueError):
            dap.build_disk_index(
                data=rando,
                distance_metric="l2",
                index_directory="test",
                complexity=5,
                graph_degree=5,
                num_of_pq_chunks=1,
                build_memory_maximum=0.01,
                num_threads=1,
                build_pq_bytes=0,
            )

        rando = rng.random(1000, dtype=np.single)
        with self.assertRaises(ValueError):
            dap.build_disk_index(
                data=rando,
                distance_metric="l2",
                index_directory="test",
                complexity=5,
                graph_degree=5,
                num_of_pq_chunks=1,
                build_memory_maximum=0.01,
                num_threads=1,
                build_pq_bytes=5
            )

        rando = rng.random(1000, dtype=np.single)
        with self.assertRaises(ValueError):
            dap.build_disk_index(
                data=rando,
                distance_metric="l2",
                index_directory="test",
                complexity=5,
                graph_degree=5,
                num_of_pq_chunks=1,
                build_memory_maximum=0.01,
                num_threads=1,
                build_pq_bytes=0
            )

        rando = rng.random(1000, dtype=np.single)
        with self.assertRaises(ValueError):
            dap.build_disk_index(
                data=rando,
                distance_metric="l2",
                index_directory="test",
                complexity=5,
                graph_degree=5,
                num_of_pq_chunks=1,
                build_memory_maximum=0.01,
                num_threads=1,
                build_pq_bytes=25
            )

    def test_value_ranges_build(self):
        good_ranges = {
            "vector_dtype": np.single,
            "distance_metric": "l2",
            "graph_degree": 5,
            "complexity": 5,
            "build_memory_maximum": 0.01,
            "num_threads": 1,
            "build_pq_bytes": 0,
            "num_of_pq_chunks": 1,
        }
        bad_ranges = {
            "vector_dtype": np.float64,
            "distance_metric": "soups this time",
            "graph_degree": -1,
            "complexity": -1,
            "build_memory_maximum": 0,
            "num_threads": -1,
            "build_pq_bytes": -1,
            "num_of_pq_chunks": 0,
        }
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest(
                f"testing bad value key: {bad_value_key} with bad value: {bad_ranges[bad_value_key]}"
            ):
                with self.assertRaises(ValueError):
                    dap.build_disk_index(
                        data="test", index_directory="test", **kwargs)


