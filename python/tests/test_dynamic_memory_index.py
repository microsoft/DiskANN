# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import shutil
import unittest

import diskannpy as dap
import numpy as np
from fixtures import build_random_vectors_and_memory_index
from sklearn.neighbors import NearestNeighbors


def _calculate_recall(
        result_set_tags: np.ndarray,
        original_indices_to_tags: np.ndarray,
        truth_set_indices: np.ndarray,
        recall_at: int = 5
) -> float:

    found = 0
    for i in range(0, result_set_tags.shape[0]):
        result_set_set = set(result_set_tags[i][0:recall_at])
        truth_set_set = set()
        for knn_index in truth_set_indices[i][0:recall_at]:
            truth_set_set.add(original_indices_to_tags[knn_index])  # mapped into our tag number instead
        found += len(result_set_set.intersection(truth_set_set))
    return found / (result_set_tags.shape[0] * recall_at)


class TestDynamicMemoryIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._test_matrix = [
            build_random_vectors_and_memory_index(np.single, "l2", with_tags=True),
            build_random_vectors_and_memory_index(np.ubyte, "l2", with_tags=True),
            build_random_vectors_and_memory_index(np.byte, "l2", with_tags=True),
            build_random_vectors_and_memory_index(np.single, "cosine", with_tags=True),
            build_random_vectors_and_memory_index(np.ubyte, "cosine", with_tags=True),
            build_random_vectors_and_memory_index(np.byte, "cosine", with_tags=True),
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
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            generated_tags
        ) in self._test_matrix:
            with self.subTest():
                index = dap.DynamicMemoryIndex.from_file(
                    index_directory=ann_dir,
                    max_vectors=11_000,
                    complexity=64,
                    graph_degree=32,
                    num_threads=16,
                )

                k = 5
                diskann_neighbors, diskann_distances = index.batch_search(
                    query_vectors,
                    k_neighbors=k,
                    complexity=5,
                    num_threads=16,
                )
                if metric == "l2" or metric == "cosine":
                    knn = NearestNeighbors(
                        n_neighbors=100, algorithm="auto", metric=metric
                    )
                    knn.fit(index_vectors)
                    knn_distances, knn_indices = knn.kneighbors(query_vectors)
                    recall = _calculate_recall(diskann_neighbors, generated_tags, knn_indices, k)
                    self.assertTrue(
                        recall > 0.70,
                        f"Recall [{recall}] was not over 0.7",
                    )

    def test_single(self):
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            generated_tags
        ) in self._test_matrix:
            with self.subTest():
                index = dap.DynamicMemoryIndex(
                    distance_metric="l2",
                    vector_dtype=dtype,
                    dimensions=10,
                    max_vectors=11_000,
                    complexity=64,
                    graph_degree=32,
                    num_threads=16,
                )
                index.batch_insert(vectors=index_vectors, vector_ids=generated_tags)

                k = 5
                ids, dists = index.search(query_vectors[0], k_neighbors=k, complexity=5)
                self.assertEqual(ids.shape[0], k)
                self.assertEqual(dists.shape[0], k)
                ids, dists = index.search(query_vectors[0].tolist(), k_neighbors=k, complexity=5)
                self.assertEqual(ids.shape[0], k)
                self.assertEqual(dists.shape[0], k)

    def test_valid_metric(self):
        with self.assertRaises(ValueError):
            dap.DynamicMemoryIndex(
                distance_metric="sandwich",
                vector_dtype=np.single,
                dimensions=10,
                max_vectors=11_000,
                complexity=64,
                graph_degree=32,
                num_threads=16,
            )
        with self.assertRaises(ValueError):
            dap.DynamicMemoryIndex(
                distance_metric=None,
                vector_dtype=np.single,
                dimensions=10,
                max_vectors=11_000,
                complexity=64,
                graph_degree=32,
                num_threads=16,
            )
        dap.DynamicMemoryIndex(
            distance_metric="l2",
            vector_dtype=np.single,
            dimensions=10,
            max_vectors=11_000,
            complexity=64,
            graph_degree=32,
            num_threads=16,
        )
        dap.DynamicMemoryIndex(
            distance_metric="mips",
            vector_dtype=np.single,
            dimensions=10,
            max_vectors=11_000,
            complexity=64,
            graph_degree=32,
            num_threads=16,
        )
        dap.DynamicMemoryIndex(
            distance_metric="MiPs",
            vector_dtype=np.single,
            dimensions=10,
            max_vectors=11_000,
            complexity=64,
            graph_degree=32,
            num_threads=16,
        )

    def test_valid_vector_dtype(self):
        aliases = {np.single: np.float32, np.byte: np.int8, np.ubyte: np.uint8}
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            generated_tags
        ) in self._test_matrix:
            with self.subTest():
                index = dap.DynamicMemoryIndex(
                    distance_metric="l2",
                    vector_dtype=aliases[dtype],
                    dimensions=10,
                    max_vectors=11_000,
                    complexity=64,
                    graph_degree=32,
                    num_threads=16,
                )

        invalid = [np.double, np.float64, np.ulonglong]
        for invalid_vector_dtype in invalid:
            with self.subTest():
                with self.assertRaises(ValueError, msg=invalid_vector_dtype):
                    dap.DynamicMemoryIndex(
                        distance_metric="l2",
                        vector_dtype=invalid_vector_dtype,
                        dimensions=10,
                        max_vectors=11_000,
                        complexity=64,
                        graph_degree=32,
                        num_threads=16,
                    )

    def test_value_ranges_ctor(self):
        (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            generated_tags
        ) = build_random_vectors_and_memory_index(np.single, "l2", with_tags=True, index_prefix="not_ann")
        good_ranges = {
            "distance_metric": "l2",
            "vector_dtype": np.single,
            "dimensions": 10,
            "max_vectors": 11_000,
            "complexity": 64,
            "graph_degree": 32,
            "max_occlusion_size": 10,
            "alpha": 1.2,
            "num_threads": 16,
            "filter_complexity": 10,
            "num_frozen_points": 10,
            "initial_search_complexity": 32,
            "search_threads": 0
        }

        bad_ranges = {
            "distance_metric": "l200000",
            "vector_dtype": np.double,
            "dimensions": -1,
            "max_vectors": -1,
            "complexity": 0,
            "graph_degree": 0,
            "max_occlusion_size": -1,
            "alpha": -1,
            "num_threads": -1,
            "filter_complexity": -1,
            "num_frozen_points": -1,
            "initial_search_complexity": -1,
            "search_threads": -1,
        }
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError, msg=f"expected to fail with parameter {bad_value_key}={bad_ranges[bad_value_key]}"):
                    index = dap.DynamicMemoryIndex(saturate_graph=False, **kwargs)

    def test_value_ranges_search(self):
        good_ranges = {"complexity": 5, "k_neighbors": 10}
        bad_ranges = {"complexity": -1, "k_neighbors": 0}
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest(msg=f"Test value ranges search with {kwargs=}"):
                with self.assertRaises(ValueError):
                    index = dap.DynamicMemoryIndex.from_file(
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        initial_search_complexity=32,
                        max_vectors=10001,
                        complexity=64,
                        graph_degree=32
                    )
                    index.search(query=np.array([], dtype=np.single), **kwargs)

    def test_value_ranges_batch_search(self):
        good_ranges = {
            "complexity": 5,
            "k_neighbors": 10,
            "num_threads": 5,
        }
        bad_ranges = {
            "complexity": 0,
            "k_neighbors": 0,
            "num_threads": -1,
        }
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest(msg=f"Testing value ranges batch search with {kwargs=}"):
                with self.assertRaises(ValueError):
                    index = dap.DynamicMemoryIndex.from_file(
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        initial_search_complexity=32,
                        max_vectors=10001,
                        complexity=64,
                        graph_degree=32
                    )
                    index.batch_search(
                        queries=np.array([[]], dtype=np.single), **kwargs
                    )
