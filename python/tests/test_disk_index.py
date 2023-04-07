# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import shutil
import unittest
from tempfile import mkdtemp

import diskannpy as dap
import numpy as np
from sklearn.neighbors import NearestNeighbors

from fixtures import calculate_recall, random_vectors, vectors_as_temp_file


class TestBuildIndex(unittest.TestCase):
    def test_valid_shape(self):
        rng = np.random.default_rng(12345)
        rando = rng.random((1000, 100, 5), dtype=np.single)
        with self.assertRaises(ValueError):
            dap.build_disk_index_from_vectors(
                rando, "l2", "test", 5, 5, 0.01, 0.01, 1, 0
            )

        rando = rng.random(1000, dtype=np.single)
        with self.assertRaises(ValueError):
            dap.build_disk_index_from_vectors(
                rando, "l2", "test", 5, 5, 0.01, 0.01, 1, 0
            )

    def test_value_ranges_build(self):
        good_ranges = {
            "vector_dtype": np.single,
            "metric": "l2",
            "max_degree": 5,
            "list_size": 5,
            "search_memory_maximum": 0.01,
            "build_memory_maximum": 0.01,
            "num_threads": 1,
            "pq_disk_bytes": 0,
        }
        bad_ranges = {
            "vector_dtype": np.float64,
            "metric": "soups this time",
            "max_degree": -1,
            "list_size": -1,
            "search_memory_maximum": 0,
            "build_memory_maximum": 0,
            "num_threads": -1,
            "pq_disk_bytes": -1,
        }
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest(
                    f"testing bad value key: {bad_value_key} with bad value: {bad_ranges[bad_value_key]}"
            ):
                with self.assertRaises(ValueError):
                    dap.build_disk_index_from_vector_file(
                        vector_bin_file="test", index_path="test", **kwargs
                    )


def _build_random_vectors_and_index(dtype, metric):
    query_vectors = random_vectors(1000, 10, dtype=dtype)
    index_vectors = random_vectors(10000, 10, dtype=dtype)
    with vectors_as_temp_file(index_vectors) as vector_temp:
        ann_dir = mkdtemp()
        dap.build_disk_index_from_vector_file(
            vector_bin_file=vector_temp,
            metric=metric,
            vector_dtype=dtype,
            index_path=ann_dir,
            max_degree=16,
            list_size=32,
            search_memory_maximum=0.00003,
            build_memory_maximum=1,
            num_threads=1,
            pq_disk_bytes=0,
        )
    return metric, dtype, query_vectors, index_vectors, ann_dir


class TestDiskIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._test_matrix = [
            _build_random_vectors_and_index(np.single, "l2"),
            _build_random_vectors_and_index(np.ubyte, "l2"),
            _build_random_vectors_and_index(np.byte, "l2"),
        ]
        cls._example_ann_dir = cls._test_matrix[0][4]

    @classmethod
    def tearDownClass(cls) -> None:
        for test in cls._test_matrix:
            try:
                _, _, _, _, ann_dir = test
                shutil.rmtree(ann_dir, ignore_errors=True)
            except:
                pass

    def test_recall_and_batch(self):
        for metric, dtype, query_vectors, index_vectors, ann_dir in self._test_matrix:
            with self.subTest():
                index = dap.DiskIndex(
                    metric="l2",
                    vector_dtype=dtype,
                    index_path=ann_dir,
                    num_threads=16,
                    num_nodes_to_cache=10,
                )

                k = 5
                diskann_neighbors, diskann_distances = index.batch_search(
                    query_vectors,
                    k_neighbors=k,
                    list_size=5,
                    beam_width=2,
                    num_threads=16,
                )
                if metric == "l2":
                    knn = NearestNeighbors(
                        n_neighbors=100, algorithm="auto", metric="l2"
                    )
                    knn.fit(index_vectors)
                    knn_distances, knn_indices = knn.kneighbors(query_vectors)
                    self.assertTrue(
                        calculate_recall(diskann_neighbors, knn_indices, k) > 0.70,
                        "Recall was not over 0.7",
                        )

    def test_single(self):
        for metric, dtype, query_vectors, index_vectors, ann_dir in self._test_matrix:
            with self.subTest():
                index = dap.DiskIndex(
                    metric="l2",
                    vector_dtype=dtype,
                    index_path=ann_dir,
                    num_threads=16,
                    num_nodes_to_cache=10,
                )

                k = 5
                ids, dists = index.search(
                    query_vectors[0], k_neighbors=k, list_size=5, beam_width=2
                )
                self.assertEqual(ids.shape[0], k)
                self.assertEqual(dists.shape[0], k)

    def test_valid_metric(self):
        ann_dir = self._example_ann_dir
        with self.assertRaises(ValueError):
            dap.DiskIndex(
                metric="sandwich",
                vector_dtype=np.single,
                index_path=ann_dir,
                num_threads=16,
                num_nodes_to_cache=10,
            )
        with self.assertRaises(ValueError):
            dap.DiskIndex(
                metric=None,
                vector_dtype=np.single,
                index_path=ann_dir,
                num_threads=16,
                num_nodes_to_cache=10,
            )
        dap.DiskIndex(
            metric="l2",
            vector_dtype=np.single,
            index_path=ann_dir,
            num_threads=16,
            num_nodes_to_cache=10,
        )
        dap.DiskIndex(
            metric="mips",
            vector_dtype=np.single,
            index_path=ann_dir,
            num_threads=16,
            num_nodes_to_cache=10,
        )
        dap.DiskIndex(
            metric="MiPs",
            vector_dtype=np.single,
            index_path=ann_dir,
            num_threads=16,
            num_nodes_to_cache=10,
        )

    def test_valid_vector_dtype(self):
        aliases = {np.single: np.float32, np.byte: np.int8, np.ubyte: np.uint8}
        for metric, dtype, query_vectors, index_vectors, ann_dir in self._test_matrix:
            with self.subTest():
                index = dap.DiskIndex(
                    metric="l2",
                    vector_dtype=aliases[dtype],
                    index_path=ann_dir,
                    num_threads=16,
                    num_nodes_to_cache=10,
                )
        ann_dir = self._example_ann_dir
        invalid = [np.double, np.float64, np.ulonglong, np.float16]
        for invalid_vector_dtype in invalid:
            with self.subTest():
                with self.assertRaises(ValueError):
                    dap.DiskIndex(
                        metric="l2",
                        vector_dtype=invalid_vector_dtype,
                        index_path=ann_dir,
                        num_threads=16,
                        num_nodes_to_cache=10,
                    )

    def test_value_ranges_search(self):
        good_ranges = {"list_size": 5, "k_neighbors": 10, "beam_width": 2}
        bad_ranges = {"list_size": -1, "k_neighbors": 0, "beam_width": 0}
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.DiskIndex(
                        metric="l2",
                        vector_dtype=np.single,
                        index_path=self._example_ann_dir,
                        num_threads=16,
                        num_nodes_to_cache=10,
                    )
                    index.search(query=np.array([], dtype=np.single), **kwargs)

    def test_value_ranges_batch_search(self):
        good_ranges = {
            "list_size": 5,
            "k_neighbors": 10,
            "beam_width": 2,
            "num_threads": 5,
        }
        bad_ranges = {
            "list_size": 0,
            "k_neighbors": 0,
            "beam_width": -1,
            "num_threads": -1,
        }
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.DiskIndex(
                        metric="l2",
                        vector_dtype=np.single,
                        index_path=self._example_ann_dir,
                        num_threads=16,
                        num_nodes_to_cache=10,
                    )
                    index.batch_search(
                        queries=np.array([[]], dtype=np.single), **kwargs
                    )
