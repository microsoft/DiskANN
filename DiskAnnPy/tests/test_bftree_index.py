# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import unittest

from pathlib import Path
from tempfile import mkdtemp

import diskannpy as dap
import numpy as np
from fixtures import (
    build_random_vectors_and_bftree_index,
    calculate_recall,
    build_empty_bftree_index,
    build_random_vectors_and_bftree_index_pq,
    build_on_disk_bftree_index,
    build_on_disk_bftree_index_pq
)
from fixtures import random_vectors
from sklearn.neighbors import NearestNeighbors
import asyncio

class TestAsyncIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._test_matrix = [
            build_random_vectors_and_bftree_index(np.float32, "l2"),
            build_random_vectors_and_bftree_index(np.uint8, "l2"),
            build_random_vectors_and_bftree_index(np.int8, "l2"),
            build_random_vectors_and_bftree_index(np.float32, "cosine"),
        ]
        cls._empty_index_matrix = [
            build_empty_bftree_index(np.float32, "l2"),
            build_empty_bftree_index(np.uint8, "l2"),
            build_empty_bftree_index(np.int8, "l2"),
            build_empty_bftree_index(np.float32, "cosine"),
        ]
        n_bytes = 8
        cls._test_matrix_pq = [
            build_random_vectors_and_bftree_index_pq(np.float32, "l2", query_vector_count=100, index_vector_count=10, num_pq_bytes=n_bytes),
            build_random_vectors_and_bftree_index_pq(np.uint8, "l2", query_vector_count=100, index_vector_count=10, num_pq_bytes=n_bytes),
            build_random_vectors_and_bftree_index_pq(np.int8, "l2", query_vector_count=100, index_vector_count=10, num_pq_bytes=n_bytes),
            build_random_vectors_and_bftree_index_pq(np.float32, "cosine", query_vector_count=100, index_vector_count=10, num_pq_bytes=n_bytes),
        ]

    def test_search(self):
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
                index,
            ) in self._test_matrix:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    k = 5
                    response = index.search(
                        query_vectors[0], k_value=k, l_value=50)
                    self.assertIsInstance(response, dap.QueryResponse)
                    ids, dists = response
                    self.assertEqual(ids.shape[0], k)
                    self.assertEqual(dists.shape[0], k)

    def test_average_degree(self):
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            index,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                avg_degree = index.get_average_degree()
                self.assertIsInstance(avg_degree, float)
                self.assertGreater(avg_degree, 0.0)

    def test_recall_and_batch(self):
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            index,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                k = 5
                batch_response = index.batch_search(
                    query_vectors,
                    k_value=k,
                    l_value=5,
                    num_threads=8,
                )
                self.assertIsInstance(batch_response, dap.QueryResponseBatchWithStats)

                diskann_neighbors, diskann_distances, diskann_stats = batch_response
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

    def test_batch_range_search(self):
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
            index,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                batch_response = index.batch_range_search(
                    query_vectors,
                    num_threads=1,
                    starting_l_value=5,
                    radius=0.5,
                )
                self.assertIsInstance(batch_response, dap.RangeQueryResponseBatchWithStats)

    def test_insert_delete(self):
        for (
            query_vectors,
            metric,
            dtype,
            index,
        ) in self._empty_index_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                for i in range(3):
                    index.insert(query_vectors[i], i)
                response = index.search(query_vectors[0], k_value=3, l_value=5)

                index.batch_insert(query_vectors[3:6], np.array([3, 4, 5]), 2)
                response = index.search(query_vectors[0], k_value=6, l_value=5)

                index.multi_insert(query_vectors[6:9], np.array([6, 7, 8]))
                response = index.search(query_vectors[0], k_value=9, l_value=5)

                index.mark_deleted(np.array([0, 3, 6]))
                response = index.search(query_vectors[0], k_value=6, l_value=50)

                index.consolidate_deletes(num_tasks=1)

                response = index.search(query_vectors[0], k_value=6, l_value=50)

                index.mark_deleted(np.array([1, 5]))
                response = index.search(query_vectors[0], k_value=6, l_value=50)

                index.consolidate_simple(num_tasks=1)
                response = index.search(query_vectors[0], k_value=6, l_value=50)

                index.multi_inplace_delete(np.array([4]))
                response = index.search(query_vectors[0], k_value=6, l_value=50)

                index.batch_inplace_delete(np.array([2,7]), num_tasks=2)
                response = index.search(query_vectors[0], k_value=6, l_value=50)

    def test_search_pq(self):
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
                index,
            ) in self._test_matrix_pq:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    k = 5
                    response = index.search(
                        query_vectors[0], k_value=k, l_value=50, use_full_precision_to_search=False)
                    self.assertIsInstance(response, dap.QueryResponse)
                    ids, dists = response
                    self.assertEqual(ids.shape[0], k)
                    self.assertEqual(dists.shape[0], k)

                    response = index.batch_search(
                        query_vectors[1:3],
                        k_value=k,
                        l_value=50,
                        num_threads=2,
                        use_full_precision_to_search=False,
                    )
                    ids, dists, stats = response

                    response = index.batch_range_search(
                        query_vectors[1:3],
                        num_threads=2,
                        starting_l_value=50,
                        radius=50000000.0,
                    )
                    diskann_lims, diskann_neighbors, diskann_distances, diskann_stats = response

                    for i in range(3, 6):    
                        index.insert(query_vectors[i+10], i)
                    response = index.search(
                        query_vectors[0], k_value=6, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response
                    
                    index.batch_insert(
                        query_vectors[6+10:9+10], 
                        np.array([6,7,8]),
                        num_tasks=2
                    )
                    response = index.search(
                        query_vectors[0], k_value=9, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response

                    index.multi_insert(
                        query_vectors[6+20:9+20], 
                        np.array([6,7,8]),
                    )
                    response = index.search(
                        query_vectors[0], k_value=9, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response

                    index.mark_deleted(np.array([0, 3, 6]))
                    response = index.search(
                        query_vectors[0], k_value=9, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response

                    index.consolidate_deletes(num_tasks=2)
                    response = index.search(
                        query_vectors[0], k_value=9, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response

                    index.mark_deleted(np.array([1, 5]))
                    response = index.search(
                        query_vectors[0], k_value=9, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response

                    index.consolidate_simple(num_tasks=2)
                    response = index.search(
                        query_vectors[0], k_value=9, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response

                    index.multi_inplace_delete(np.array([4, 8]))
                    response = index.search(
                        query_vectors[0], k_value=9, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response

                    index.batch_inplace_delete(np.array([2,7]), num_tasks=2)
                    response = index.search(
                        query_vectors[0], k_value=9, l_value=50, use_full_precision_to_search=False)
                    ids, dists = response

    def test_on_disk_index(self):
        """Test that on-disk BfTree indexes create actual files and produce correct search results."""
        for dtype, metric in [
            (np.float32, "l2"),
            (np.uint8, "l2"),
            (np.int8, "l2"),
            (np.float32, "cosine"),
        ]:
            with self.subTest(msg=f"Testing on-disk dtype {dtype} metric {metric}"):
                (
                    metric_out,
                    dtype_out,
                    query_vectors,
                    index_vectors,
                    ann_dir,
                    on_disk_prefix,
                    index,
                ) = build_on_disk_bftree_index(dtype, metric)

                # Ensure cleanup happens even if assertions fail
                self.addCleanup(lambda: shutil.rmtree(ann_dir, ignore_errors=True))

                # Verify BfTree files exist on disk with non-zero size
                vectors_path = f"{on_disk_prefix}_vectors.bftree"
                neighbors_path = f"{on_disk_prefix}_neighbors.bftree"
                self.assertTrue(os.path.exists(vectors_path), f"Missing {vectors_path}")
                self.assertTrue(os.path.exists(neighbors_path), f"Missing {neighbors_path}")
                self.assertGreater(os.path.getsize(vectors_path), 0, "vectors file is empty")
                self.assertGreater(os.path.getsize(neighbors_path), 0, "neighbors file is empty")

                # Verify search works
                k = 5
                response = index.search(query_vectors[0], k_value=k, l_value=50)
                self.assertIsInstance(response, dap.QueryResponse)
                ids, dists = response
                self.assertEqual(ids.shape[0], k)
                self.assertEqual(dists.shape[0], k)

                # Verify batch search works
                batch_response = index.batch_search(
                    query_vectors[:10],
                    k_value=k,
                    l_value=50,
                    num_threads=2,
                )
                self.assertIsInstance(batch_response, dap.QueryResponseBatchWithStats)

                # Verify recall is reasonable
                if metric in ["l2", "cosine"]:
                    diskann_neighbors, diskann_distances, _ = batch_response
                    knn = NearestNeighbors(
                        n_neighbors=100, algorithm="auto", metric=metric
                    )
                    knn.fit(index_vectors)
                    knn_distances, knn_indices = knn.kneighbors(query_vectors[:10])
                    recall = calculate_recall(diskann_neighbors, knn_indices, k)
                    self.assertTrue(
                        recall > 0.70,
                        f"Recall [{recall}] was not over 0.7 for dtype {dtype}",
                    )

    def test_on_disk_index_pq(self):
        """Test that on-disk PQ BfTree indexes create actual files and produce correct search results."""
        n_bytes = 8
        for dtype, metric in [
            (np.float32, "l2"),
            (np.uint8, "l2"),
            (np.int8, "l2"),
            (np.float32, "cosine"),
        ]:
            with self.subTest(msg=f"Testing on-disk PQ dtype {dtype} metric {metric}"):
                (
                    metric_out,
                    dtype_out,
                    query_vectors,
                    index_vectors,
                    ann_dir,
                    on_disk_prefix,
                    index,
                ) = build_on_disk_bftree_index_pq(dtype, metric, num_pq_bytes=n_bytes)

                # Ensure cleanup happens even if assertions fail
                self.addCleanup(lambda: shutil.rmtree(ann_dir, ignore_errors=True))

                # Verify BfTree files exist on disk with non-zero size
                vectors_path = f"{on_disk_prefix}_vectors.bftree"
                neighbors_path = f"{on_disk_prefix}_neighbors.bftree"
                quant_path = f"{on_disk_prefix}_quant.bftree"
                self.assertTrue(os.path.exists(vectors_path), f"Missing {vectors_path}")
                self.assertTrue(os.path.exists(neighbors_path), f"Missing {neighbors_path}")
                self.assertTrue(os.path.exists(quant_path), f"Missing {quant_path}")
                self.assertGreater(os.path.getsize(vectors_path), 0, "vectors file is empty")
                self.assertGreater(os.path.getsize(neighbors_path), 0, "neighbors file is empty")
                self.assertGreater(os.path.getsize(quant_path), 0, "quant file is empty")

                # Verify search works with PQ
                k = 5
                response = index.search(
                    query_vectors[0], k_value=k, l_value=50, use_full_precision_to_search=False)
                self.assertIsInstance(response, dap.QueryResponse)
                ids, dists = response
                self.assertEqual(ids.shape[0], k)
                self.assertEqual(dists.shape[0], k)

                # Verify batch search works
                batch_response = index.batch_search(
                    query_vectors[1:3],
                    k_value=k,
                    l_value=50,
                    num_threads=2,
                    use_full_precision_to_search=False,
                )
                ids, dists, stats = batch_response

                # Verify batch range search works
                response = index.batch_range_search(
                    query_vectors[1:3],
                    num_threads=2,
                    starting_l_value=50,
                    radius=50000000.0,
                )
                self.assertIsInstance(response, dap.RangeQueryResponseBatchWithStats)

    def test_save_load(self):
        """Test saving an on-disk index, loading it back, and verifying search results match."""
        for dtype, metric in [
            (np.float32, "l2"),
            (np.uint8, "l2"),
            (np.int8, "l2"),
            (np.float32, "cosine"),
        ]:
            with self.subTest(msg=f"Testing save/load dtype {dtype} metric {metric}"):
                (
                    metric_out,
                    dtype_out,
                    query_vectors,
                    index_vectors,
                    ann_dir,
                    on_disk_prefix,
                    index,
                ) = build_on_disk_bftree_index(dtype, metric)

                # Ensure cleanup happens even if assertions fail
                self.addCleanup(lambda: shutil.rmtree(ann_dir, ignore_errors=True))

                # Search on the original index
                k = 5
                original_response = index.search(query_vectors[0], k_value=k, l_value=50)
                original_ids, original_dists = original_response

                # Save the index (uses the same on_disk_prefix)
                index.save(on_disk_prefix)

                # Verify params JSON was created
                self.assertTrue(
                    os.path.exists(f"{on_disk_prefix}_params.json"),
                    "params.json not created by save",
                )

                # Load the index
                loaded_index = dap.BfTreeIndex.load(
                    prefix=on_disk_prefix,
                )

                # Search on the loaded index
                loaded_response = loaded_index.search(query_vectors[0], k_value=k, l_value=50)
                loaded_ids, loaded_dists = loaded_response

                # Verify results match
                self.assertEqual(loaded_ids.shape[0], k)
                self.assertEqual(loaded_dists.shape[0], k)
                np.testing.assert_array_equal(original_ids, loaded_ids)
                np.testing.assert_array_almost_equal(original_dists, loaded_dists)

                # Verify batch search works on loaded index
                batch_response = loaded_index.batch_search(
                    query_vectors[:5],
                    k_value=k,
                    l_value=50,
                    num_threads=2,
                )
                self.assertIsInstance(batch_response, dap.QueryResponseBatchWithStats)

                # Verify average degree is preserved
                original_avg = index.get_average_degree()
                loaded_avg = loaded_index.get_average_degree()
                self.assertAlmostEqual(original_avg, loaded_avg, places=5)

    def test_save_load_pq(self):
        """Test saving and loading an on-disk PQ index."""
        n_bytes = 8
        for dtype, metric in [
            (np.float32, "l2"),
            (np.uint8, "l2"),
            (np.int8, "l2"),
            (np.float32, "cosine"),
        ]:
            with self.subTest(msg=f"Testing save/load PQ dtype {dtype} metric {metric}"):
                (
                    metric_out,
                    dtype_out,
                    query_vectors,
                    index_vectors,
                    ann_dir,
                    on_disk_prefix,
                    index,
                ) = build_on_disk_bftree_index_pq(dtype, metric, num_pq_bytes=n_bytes)

                # Ensure cleanup happens even if assertions fail
                self.addCleanup(lambda: shutil.rmtree(ann_dir, ignore_errors=True))

                # Search on the original PQ index
                k = 5
                original_response = index.search(
                    query_vectors[0], k_value=k, l_value=50, use_full_precision_to_search=False
                )
                original_ids, original_dists = original_response

                # Save the index
                index.save(on_disk_prefix)

                # Verify params JSON was created
                self.assertTrue(
                    os.path.exists(f"{on_disk_prefix}_params.json"),
                    "params.json not created by save",
                )

                # Load the index
                loaded_index = dap.BfTreeIndex.load(
                    prefix=on_disk_prefix,
                )

                # Search on the loaded PQ index
                loaded_response = loaded_index.search(
                    query_vectors[0], k_value=k, l_value=50, use_full_precision_to_search=False
                )
                loaded_ids, loaded_dists = loaded_response

                # Verify results match
                self.assertEqual(loaded_ids.shape[0], k)
                self.assertEqual(loaded_dists.shape[0], k)
                np.testing.assert_array_equal(original_ids, loaded_ids)
                np.testing.assert_array_almost_equal(original_dists, loaded_dists)

                # Verify batch search works on loaded PQ index
                batch_response = loaded_index.batch_search(
                    query_vectors[1:3],
                    k_value=k,
                    l_value=50,
                    num_threads=2,
                    use_full_precision_to_search=False,
                )
                ids, dists, stats = batch_response

    @classmethod
    def tearDownClass(cls) -> None:
        for test in cls._test_matrix:
            try:
                ann_dir = test[4]
                shutil.rmtree(ann_dir, ignore_errors=True)
            except:
                pass