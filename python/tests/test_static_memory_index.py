# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import shutil
import unittest

import diskannpy as dap
import numpy as np
from fixtures import build_random_vectors_and_memory_index, calculate_recall
from sklearn.neighbors import NearestNeighbors


class TestStaticMemoryIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._test_matrix = [
            build_random_vectors_and_memory_index(np.single, "l2"),
            build_random_vectors_and_memory_index(np.ubyte, "l2"),
            build_random_vectors_and_memory_index(np.byte, "l2"),
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
            _
        ) in self._test_matrix:
            with self.subTest():
                index = dap.StaticMemoryIndex(
                    metric="l2",
                    vector_dtype=dtype,
                    data_path=os.path.join(ann_dir, "vectors.bin"),
                    index_directory=ann_dir,
                    num_threads=16,
                    initial_search_complexity=32,
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
                    recall = calculate_recall(diskann_neighbors, knn_indices, k)
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
            _
        ) in self._test_matrix:
            with self.subTest():
                index = dap.StaticMemoryIndex(
                    metric="l2",
                    vector_dtype=dtype,
                    data_path=vector_bin_file,
                    index_directory=ann_dir,
                    num_threads=16,
                    initial_search_complexity=32,
                )

                k = 5
                ids, dists = index.search(query_vectors[0], k_neighbors=k, complexity=5)
                self.assertEqual(ids.shape[0], k)
                self.assertEqual(dists.shape[0], k)

    def test_valid_metric(self):
        ann_dir = self._example_ann_dir
        vector_bin_file = self._test_matrix[0][5]
        with self.assertRaises(ValueError):
            dap.StaticMemoryIndex(
                metric="sandwich",
                vector_dtype=np.single,
                data_path=vector_bin_file,
                index_directory=ann_dir,
                num_threads=16,
                initial_search_complexity=32,
            )
        with self.assertRaises(ValueError):
            dap.StaticMemoryIndex(
                metric=None,
                vector_dtype=np.single,
                data_path=vector_bin_file,
                index_directory=ann_dir,
                num_threads=16,
                initial_search_complexity=32,
            )
        dap.StaticMemoryIndex(
            metric="l2",
            vector_dtype=np.single,
            data_path=vector_bin_file,
            index_directory=ann_dir,
            num_threads=16,
            initial_search_complexity=32,
        )
        dap.StaticMemoryIndex(
            metric="mips",
            vector_dtype=np.single,
            data_path=vector_bin_file,
            index_directory=ann_dir,
            num_threads=16,
            initial_search_complexity=32,
        )
        dap.StaticMemoryIndex(
            metric="MiPs",
            vector_dtype=np.single,
            data_path=vector_bin_file,
            index_directory=ann_dir,
            num_threads=16,
            initial_search_complexity=32,
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
            _
        ) in self._test_matrix:
            with self.subTest():
                index = dap.StaticMemoryIndex(
                    metric="l2",
                    vector_dtype=aliases[dtype],
                    data_path=vector_bin_file,
                    index_directory=ann_dir,
                    num_threads=16,
                    initial_search_complexity=32,
                )
        ann_dir = self._example_ann_dir
        vector_bin_file = self._test_matrix[0][5]
        invalid = [np.double, np.float64, np.ulonglong, np.float16]
        for invalid_vector_dtype in invalid:
            with self.subTest():
                with self.assertRaises(ValueError):
                    dap.StaticMemoryIndex(
                        metric="l2",
                        vector_dtype=invalid_vector_dtype,
                        data_path=vector_bin_file,
                        index_directory=ann_dir,
                        num_threads=16,
                        initial_search_complexity=32,
                    )

    def test_value_ranges_ctor(self):
        (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            _
        ) = build_random_vectors_and_memory_index(np.single, "l2", "not_ann")
        good_ranges = {
            "metric": "l2",
            "vector_dtype": np.single,
            "data_path": vector_bin_file,
            "index_directory": ann_dir,
            "num_threads": 16,
            "initial_search_complexity": 32,
            "index_prefix": "not_ann",
        }

        bad_ranges = {
            "metric": "l200000",
            "vector_dtype": np.double,
            "data_path": "I do not exist.bin",
            "index_directory": "sandwiches",
            "num_threads": -100,
            "initial_search_complexity": 0,
            "index_prefix": "",
        }
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.StaticMemoryIndex(**kwargs)

    def test_value_ranges_search(self):
        good_ranges = {"complexity": 5, "k_neighbors": 10}
        bad_ranges = {"complexity": -1, "k_neighbors": 0}
        vector_bin_file = self._test_matrix[0][5]
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.StaticMemoryIndex(
                        metric="l2",
                        vector_dtype=np.single,
                        data_path=vector_bin_file,
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        initial_search_complexity=32,
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
        vector_bin_file = self._test_matrix[0][5]
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.StaticMemoryIndex(
                        metric="l2",
                        vector_dtype=np.single,
                        data_path=vector_bin_file,
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        initial_search_complexity=32,
                    )
                    index.batch_search(
                        queries=np.array([[]], dtype=np.single), **kwargs
                    )
