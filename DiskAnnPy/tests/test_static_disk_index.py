# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import time
import unittest

from pathlib import Path
from tempfile import mkdtemp

import diskannpy as dap
import numpy as np
from fixtures import build_random_vectors_and_disk_index, calculate_recall, PQ_RECALL_CUTOFF
from fixtures import random_vectors
from sklearn.neighbors import NearestNeighbors


class TestStaticDiskIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._test_matrix = [
            build_random_vectors_and_disk_index(np.float32, "l2"),
            build_random_vectors_and_disk_index(np.uint8, "l2"),
            build_random_vectors_and_disk_index(np.int8, "l2"),
            build_random_vectors_and_disk_index(np.float32, "cosine"),
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
            _,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.StaticDiskIndex(
                    index_directory=ann_dir,
                    num_threads=16,
                    beam_width=2,
                )

                k = 5
                batch_response = index.batch_search(
                    query_vectors,
                    k_neighbors=k,
                    complexity=6,
                    num_threads=16,
                )
                self.assertIsInstance(batch_response, dap.QueryResponseBatchWithStats)

                diskann_neighbors, diskann_distances, diskann_stats = batch_response
                if metric in ["l2", "cosine"]:
                    knn = NearestNeighbors(
                        n_neighbors=100, algorithm="auto", metric=metric
                    )
                    knn.fit(index_vectors)
                    knn_distances, knn_indices = knn.kneighbors(query_vectors)
                    recall = calculate_recall(
                        diskann_neighbors, knn_indices, k)
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
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.StaticDiskIndex(
                    index_directory=ann_dir,
                    num_threads=16,
                    beam_width=2,
                )

                k = 5
                response = index.search(
                    query_vectors[0], k_neighbors=k, complexity=5)
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
        ) = build_random_vectors_and_disk_index(np.single, "l2", "not_ann")
        good_ranges = {
            "index_directory": ann_dir,
            "num_threads": 16,
            "index_prefix": "not_ann",
            "beam_width": 2,
        }

        bad_ranges = {
            "index_directory": "sandwiches",
            "num_threads": -100,
            "index_prefix": "",
            "beam_width": -1,
        }
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.StaticDiskIndex(**kwargs)

    def test_value_ranges_search(self):
        good_ranges = {"complexity": 5, "k_neighbors": 10}
        bad_ranges = {"complexity": -1, "k_neighbors": 0}
        for bad_value_key in good_ranges.keys():
            kwargs = good_ranges.copy()
            kwargs[bad_value_key] = bad_ranges[bad_value_key]
            with self.subTest():
                with self.assertRaises(ValueError):
                    index = dap.StaticDiskIndex(
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        beam_width=2,
                    )
                    index.search(query=np.array(
                        [], dtype=np.single), **kwargs)

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
                    index = dap.StaticDiskIndex(
                        index_directory=self._example_ann_dir,
                        num_threads=16,
                        beam_width=2,
                    )
                    index.batch_search(
                        queries=np.array([[]], dtype=np.single), **kwargs
                    )

    def test_relative_paths(self):
        # Issue 483 and 491 both fixed errors that were somehow slipping past our unit tests
        # os.path.join() acts as a semi-merge if you give it two paths that look absolute.
        # since our unit tests are using absolute paths via tempfile.mkdtemp(), the double os.path.join() was never
        # caught by our tests, but was very easy to trip when using relative paths
        rel_dir = "tmp"
        Path(rel_dir).mkdir(exist_ok=True)
        try:
            tiny_index_vecs = random_vectors(
                20, 10, dtype=np.float32, seed=12345)
            dap.build_disk_index(
                data=tiny_index_vecs,
                distance_metric="l2",
                index_directory=rel_dir,
                graph_degree=16,
                complexity=32,
                build_memory_maximum=1,
                num_threads=0,
                num_of_pq_chunks=10,
            )
            index = dap.StaticDiskIndex(
                index_directory=rel_dir,
                num_threads=0,
                beam_width=2,
            )

        finally:
            shutil.rmtree(rel_dir, ignore_errors=True)
