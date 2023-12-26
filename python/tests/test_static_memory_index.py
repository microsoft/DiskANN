# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import shutil
import unittest

from pathlib import Path
from tempfile import mkdtemp

import diskannpy as dap
import numpy as np
from fixtures import build_random_vectors_and_memory_index, calculate_recall
from fixtures import  random_vectors
from sklearn.neighbors import NearestNeighbors


class TestStaticMemoryIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._test_matrix = [
            build_random_vectors_and_memory_index(np.float32, "l2"),
            build_random_vectors_and_memory_index(np.uint8, "l2"),
            build_random_vectors_and_memory_index(np.int8, "l2"),
            build_random_vectors_and_memory_index(np.float32, "cosine"),
            build_random_vectors_and_memory_index(np.uint8, "cosine"),
            build_random_vectors_and_memory_index(np.int8, "cosine"),
            build_random_vectors_and_memory_index(np.float32, "mips"),
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
            _,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.StaticMemoryIndex(
                    index_directory=ann_dir,
                    num_threads=16,
                    initial_search_complexity=32,
                )

                k = 5
                batch_response = index.batch_search(
                    query_vectors,
                    k_neighbors=k,
                    complexity=5,
                    num_threads=16,
                )
                self.assertIsInstance(batch_response, dap.QueryResponseBatch)

                diskann_neighbors, diskann_distances = batch_response
                if metric in ["l2", "cosine"]:
                    knn = NearestNeighbors(
                        n_neighbors=100, algorithm="auto", metric=metric
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
            _,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.StaticMemoryIndex(
                    index_directory=ann_dir,
                    num_threads=16,
                    initial_search_complexity=32,
                )

                k = 5
                response = index.search(query_vectors[0], k_neighbors=k, complexity=5)
                self.assertIsInstance(response, dap.QueryResponse)
                ids, dists = response
                self.assertEqual(ids.shape[0], k)
                self.assertEqual(dists.shape[0], k)

    def test_value_ranges_ctor(self):
        (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            _,
        ) = build_random_vectors_and_memory_index(np.single, "l2", "not_ann")
        good_ranges = {
            "index_directory": ann_dir,
            "num_threads": 16,
            "initial_search_complexity": 32,
            "index_prefix": "not_ann",
        }

        bad_ranges = {
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
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.StaticMemoryIndex(
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
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        initial_search_complexity=32,
                    )
                    index.batch_search(
                        queries=np.array([[]], dtype=np.single), **kwargs
                    )

    def test_zero_threads(self):
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            _,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.StaticMemoryIndex(
                    index_directory=ann_dir,
                    num_threads=0,
                    initial_search_complexity=32,
                )

                k = 5
                ids, dists = index.batch_search(query_vectors, k_neighbors=k, complexity=5, num_threads=0)

    def test_relative_paths(self):
        # Issue 483 and 491 both fixed errors that were somehow slipping past our unit tests
        # os.path.join() acts as a semi-merge if you give it two paths that look absolute.
        # since our unit tests are using absolute paths via tempfile.mkdtemp(), the double os.path.join() was never
        # caught by our tests, but was very easy to trip when using relative paths
        rel_dir = "tmp"
        Path(rel_dir).mkdir(exist_ok=True)
        try:
            tiny_index_vecs = random_vectors(20, 10, dtype=np.float32, seed=12345)
            dap.build_memory_index(
                data=tiny_index_vecs,
                distance_metric="l2",
                index_directory=rel_dir,
                graph_degree=16,
                complexity=32,
                num_threads=0,
            )
            index = dap.StaticMemoryIndex(
                index_directory=rel_dir,
                num_threads=0,
                initial_search_complexity=32,
            )

        finally:
            shutil.rmtree(rel_dir, ignore_errors=True)



class TestFilteredStaticMemoryIndex(unittest.TestCase):
    def test_simple_scenario(self):
        vectors: np.ndarray = random_vectors(10000, 10, dtype=np.float32, seed=54321)
        query_vectors: np.ndarray = random_vectors(10, 10, dtype=np.float32)
        temp = mkdtemp()
        labels = []
        for idx in range(0, vectors.shape[0]):
            label_list = []
            if idx % 3 == 0:
                label_list.append("even_by_3")
            if idx % 5 == 0:
                label_list.append("even_by_5")
            if len(label_list) == 0:
                label_list = ["neither"]
            labels.append(label_list)
        try:
            dap.build_memory_index(
                data=vectors,
                distance_metric="l2",
                index_directory=temp,
                complexity=64,
                graph_degree=32,
                num_threads=16,
                filter_labels=labels,
                universal_label="all",
                filter_complexity=128,
            )
            index = dap.StaticMemoryIndex(
                index_directory=temp,
                num_threads=16,
                initial_search_complexity=64,
                enable_filters=True
            )

            k = 50
            probable_superset, _ = index.search(query_vectors[0], k_neighbors=k*2, complexity=128)
            ids_1, _ = index.search(query_vectors[0], k_neighbors=k, complexity=64, filter_label="even_by_3")
            self.assertTrue(all(id % 3 == 0 for id in ids_1))
            ids_2, _ = index.search(query_vectors[0], k_neighbors=k, complexity=64, filter_label="even_by_5")
            self.assertTrue(all(id % 5 == 0 for id in ids_2))

            in_superset = np.intersect1d(probable_superset, np.append(ids_1, ids_2)).shape[0]
            self.assertTrue(in_superset/k*2 > 0.98)
        finally:
            shutil.rmtree(temp, ignore_errors=True)


    def test_exhaustive_validation(self):
        vectors: np.ndarray = random_vectors(10000, 10, dtype=np.float32, seed=54321)
        query_vectors: np.ndarray = random_vectors(10, 10, dtype=np.float32)
        temp = mkdtemp()
        labels = []
        for idx in range(0, vectors.shape[0]):
            label_list = []
            label_list.append("all")
            if idx % 2 == 0:
                label_list.append("even")
            else:
                label_list.append("odd")
            if idx % 3 == 0:
                label_list.append("by_three")
            labels.append(label_list)
        try:
            dap.build_memory_index(
                data=vectors,
                distance_metric="l2",
                index_directory=temp,
                complexity=64,
                graph_degree=32,
                num_threads=16,
                filter_labels=labels,
                universal_label="",
                filter_complexity=128,
            )
            index = dap.StaticMemoryIndex(
                index_directory=temp,
                num_threads=16,
                initial_search_complexity=64,
                enable_filters=True
            )

            k = 5_000
            without_filter, _ = index.search(query_vectors[0], k_neighbors=k*2, complexity=128)
            with_filter_but_label_all, _ = index.search(
                query_vectors[0], k_neighbors=k*2, complexity=128, filter_label="all"
            )
            intersection = np.intersect1d(without_filter, with_filter_but_label_all)
            intersect_count = intersection.shape[0]
            self.assertEqual(intersect_count, k*2)

            ids_1, _ = index.search(query_vectors[0], k_neighbors=k*10, complexity=128, filter_label="even")
            # we ask for more than 5000. prior to the addition of the `_label_metadata.json` file
            # asking for more k than we had items with that label would result in nonsense results past the first
            # 5000.
            self.assertTrue(all(id % 2 == 0 for id in ids_1))
            ids_2, _ = index.search(query_vectors[0], k_neighbors=k, complexity=128, filter_label="odd")
            self.assertTrue(all(id % 2 != 0 for id in ids_2))

        finally:
            shutil.rmtree(temp, ignore_errors=True)
