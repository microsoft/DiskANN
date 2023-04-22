# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import shutil
import unittest
from tempfile import mkdtemp

import diskannpy as dap
import numpy as np
from fixtures import calculate_recall, random_vectors, vectors_as_temp_file
from sklearn.neighbors import NearestNeighbors


def _build_random_vectors_and_index(dtype, metric):
    query_vectors: np.ndarray = random_vectors(1000, 10, dtype=dtype)
    index_vectors: np.ndarray = random_vectors(10000, 10, dtype=dtype)
    ann_dir = mkdtemp()
    dap.build_memory_index(
        data=index_vectors,
        metric=metric,
        index_directory=ann_dir,
        graph_degree=16,
        complexity=32,
        alpha=1.2,
        num_threads=0,
        use_pq_build=False,
        num_pq_bytes=8,
        use_opq=False,
        filter_complexity=32
    )
    return metric, dtype, query_vectors, index_vectors, ann_dir


class TestStaticMemoryIndex(unittest.TestCase):
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
                ann_dir = test[4]
                shutil.rmtree(ann_dir, ignore_errors=True)
            except:
                pass

    def test_recall_and_batch(self):
        for metric, dtype, query_vectors, index_vectors, ann_dir in self._test_matrix:
            with self.subTest():
                index = dap.StaticMemoryIndex(
                    metric="l2",
                    vector_dtype=dtype,
                    data_path=os.path.join(ann_dir, "vectors.bin"),
                    index_directory=ann_dir,
                    num_threads=16,
                    initial_search_complexity=32
                )

                k = 5
                diskann_neighbors, diskann_distances = index.batch_search(
                    query_vectors,
                    k_neighbors=k,
                    complexity=5,
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
                index = dap.StaticMemoryIndex(
                    metric="l2",
                    vector_dtype=dtype,
                    data_path=os.path.join(ann_dir, "vectors.bin"),
                    index_directory=ann_dir,
                    num_threads=16,
                    initial_search_complexity=32
                )

                k = 5
                ids, dists = index.search(
                    query_vectors[0], k_neighbors=k, complexity=5
                )
                self.assertEqual(ids.shape[0], k)
                self.assertEqual(dists.shape[0], k)

    @unittest.SkipTest
    def test_valid_metric(self):
        ann_dir = self._example_ann_dir
        with self.assertRaises(ValueError):
            dap.DiskIndex(
                metric="sandwich",
                vector_dtype=np.single,
                index_directory=ann_dir,
                num_threads=16,
                num_nodes_to_cache=10,
            )
        with self.assertRaises(ValueError):
            dap.DiskIndex(
                metric=None,
                vector_dtype=np.single,
                index_directory=ann_dir,
                num_threads=16,
                num_nodes_to_cache=10,
            )
        dap.DiskIndex(
            metric="l2",
            vector_dtype=np.single,
            index_directory=ann_dir,
            num_threads=16,
            num_nodes_to_cache=10,
        )
        dap.DiskIndex(
            metric="mips",
            vector_dtype=np.single,
            index_directory=ann_dir,
            num_threads=16,
            num_nodes_to_cache=10,
        )
        dap.DiskIndex(
            metric="MiPs",
            vector_dtype=np.single,
            index_directory=ann_dir,
            num_threads=16,
            num_nodes_to_cache=10,
        )

    @unittest.SkipTest
    def test_valid_vector_dtype(self):
        aliases = {np.single: np.float32, np.byte: np.int8, np.ubyte: np.uint8}
        for metric, dtype, query_vectors, index_vectors, ann_dir in self._test_matrix:
            with self.subTest():
                index = dap.DiskIndex(
                    metric="l2",
                    vector_dtype=aliases[dtype],
                    index_directory=ann_dir,
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
                        index_directory=ann_dir,
                        num_threads=16,
                        num_nodes_to_cache=10,
                    )

    @unittest.SkipTest
    def test_value_ranges_search(self):
        good_ranges = {"complexity": 5, "k_neighbors": 10, "beam_width": 2}
        bad_ranges = {"complexity": -1, "k_neighbors": 0, "beam_width": 0}
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.DiskIndex(
                        metric="l2",
                        vector_dtype=np.single,
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        num_nodes_to_cache=10,
                    )
                    index.search(query=np.array([], dtype=np.single), **kwargs)

    @unittest.SkipTest
    def test_value_ranges_batch_search(self):
        good_ranges = {
            "complexity": 5,
            "k_neighbors": 10,
            "beam_width": 2,
            "num_threads": 5,
        }
        bad_ranges = {
            "complexity": 0,
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
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        num_nodes_to_cache=10,
                    )
                    index.batch_search(
                        queries=np.array([[]], dtype=np.single), **kwargs
                    )
